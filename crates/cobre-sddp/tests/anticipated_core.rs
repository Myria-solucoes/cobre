//! Consolidated anticipated-core integration tests for `cobre-sddp`.
//!
//! Each anticipated-core domain group lives in its own inner `mod` so the suite
//! links the statically-bound solver once rather than once per file. Per-`mod`
//! scoping isolates each group's consts, helpers, and fixtures.

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

mod anticipated_backward_cut {
    //! Analytical verification of backward-pass cut-coefficient extraction for an
    //! anticipated thermal across lead_stages K = 1, 2, 3. Each K's closed-form
    //! derivation lives on its test function.

    use cobre_core::entities::{bus::DeficitSegment, thermal::AnticipatedConfig};
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroStageBounds, HydroStagePenalties,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
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
    // Numeric constants shared across all K (single source of truth).
    // ---------------------------------------------------------------------------

    const BLOCK_HOURS: f64 = 1.0;
    const C_REG: f64 = 100.0;
    const C_ANT: f64 = 50.0;

    // Every non-theta objective coefficient is divided by this, so duals and the
    // stored cut live in scaled cost units.
    const COST_SCALE_FACTOR: f64 = 1_000_000.0;

    const TOL: f64 = 1e-6;

    // System::build sorts thermals by EntityId ascending; with reg_id < ant_id,
    // thermal_idx 0 is the regular thermal and 1 is the anticipated thermal.
    const THERMAL_IDX_REG: usize = 0;
    const THERMAL_IDX_ANT: usize = 1;

    // ---------------------------------------------------------------------------
    // Per-K fixture table
    // ---------------------------------------------------------------------------

    /// Per-K parameters for the anticipated backward-cut fixtures. Each `#[test]`
    /// builds an independent `System` from one entry, so entity IDs need only be
    /// disjoint within an entry (bus 1, reg, ant); cross-K reuse is harmless.
    struct BackwardCutFixture {
        n_stages: usize,
        k_max: usize,
        /// Per-stage load, MW (length `n_stages`).
        loads_mw: &'static [f64],
        max_gen_reg: f64,
        max_gen_ant: f64,
        reg_id: EntityId,
        ant_id: EntityId,
        reg_start_date: (i32, u32, u32),
        ant_start_date: (i32, u32, u32),
        /// Anticipated ring-buffer seeds, MW (length `k_max`).
        seeds_mw: &'static [f64],
        iterations: usize,
        expected_coeff: f64,
    }

    const FIXTURE_K1: BackwardCutFixture = BackwardCutFixture {
        n_stages: 2,
        k_max: 1,
        loads_mw: &[10.0, 20.0],
        max_gen_reg: 50.0,
        max_gen_ant: 30.0,
        reg_id: EntityId(3),
        ant_id: EntityId(4),
        reg_start_date: (2024, 1, 4),
        ant_start_date: (2024, 1, 5),
        seeds_mw: &[10.0],
        iterations: 1,
        expected_coeff: -C_REG / COST_SCALE_FACTOR,
    };

    const FIXTURE_K2: BackwardCutFixture = BackwardCutFixture {
        n_stages: 3,
        k_max: 2,
        loads_mw: &[5.0, 10.0, 30.0],
        max_gen_reg: 100.0,
        max_gen_ant: 50.0,
        reg_id: EntityId(2),
        ant_id: EntityId(4),
        reg_start_date: (2024, 1, 3),
        ant_start_date: (2024, 1, 5),
        seeds_mw: &[0.0, 0.0],
        iterations: 5,
        expected_coeff: -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS,
    };

    const FIXTURE_K3: BackwardCutFixture = BackwardCutFixture {
        n_stages: 4,
        k_max: 3,
        loads_mw: &[5.0, 10.0, 15.0, 30.0],
        max_gen_reg: 100.0,
        max_gen_ant: 50.0,
        reg_id: EntityId(2),
        ant_id: EntityId(5),
        reg_start_date: (2024, 1, 3),
        ant_start_date: (2024, 1, 6),
        seeds_mw: &[0.0, 0.0, 0.0],
        iterations: 5,
        expected_coeff: -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS,
    };

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    fn build_system(fixture: &BackwardCutFixture) -> cobre_core::System {
        use chrono::NaiveDate;

        let date = |d: (i32, u32, u32)| NaiveDate::from_ymd_opt(d.0, d.1, d.2).expect("valid date");

        let bus = make_bus(
            EntityId(1),
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 2).unwrap(),
                // Deficit cost set safely above c_reg so the LP never prefers shedding load.
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 1000.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let thermal_reg = make_thermal(
            fixture.reg_id,
            ThermalSpec {
                name: "T_reg".to_string(),
                operational_start_date: date(fixture.reg_start_date),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: fixture.max_gen_reg,
                cost_per_mwh: C_REG,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_ant = make_thermal(
            fixture.ant_id,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: date(fixture.ant_start_date),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: fixture.max_gen_ant,
                cost_per_mwh: C_ANT,
                anticipated_config: Some(AnticipatedConfig::LeadStages(fixture.k_max as u32)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        assert!(
            thermal_reg.id.0 < thermal_ant.id.0,
            "R7: T_reg.id ({}) must be strictly less than T_ant.id ({}) so that \
         System::build's sort_by_key aligns thermal_idx with the bounds table",
            thermal_reg.id.0,
            thermal_ant.id.0,
        );

        let stages: Vec<Stage> = (0..fixture.n_stages)
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

        let load_models: Vec<LoadModel> = (0..fixture.n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: fixture.loads_mw[i],
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
                n_stages: fixture.n_stages,
                k_max: fixture.k_max,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 0.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                    max_diversion_m3s: None,
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
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
        );

        // K-padded axis: fill_anticipated_columns reads delivery cells at
        // stage_idx + K_i, so overrides must cover the n_stages + k_max range.
        let thermal_axis = fixture.n_stages + fixture.k_max;
        for s in 0..thermal_axis {
            *bounds.thermal_bounds_mut(THERMAL_IDX_REG, s) = ThermalStageBounds {
                cost_per_mwh: C_REG,
            };
            *bounds.thermal_block_base_mut(THERMAL_IDX_REG, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: fixture.max_gen_reg,
            };
            *bounds.thermal_bounds_mut(THERMAL_IDX_ANT, s) = ThermalStageBounds {
                cost_per_mwh: C_ANT,
            };
            *bounds.thermal_block_base_mut(THERMAL_IDX_ANT, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: fixture.max_gen_ant,
            };
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages: fixture.n_stages,
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

        // Anticipated ring-buffer seeds; any feasible choice yields the same cut.
        let initial_conditions = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: fixture.ant_id,
                values_mw: fixture.seeds_mw.to_vec(),
            }],
            recent_observations: vec![],
            past_defluences: vec![],
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

    fn build_config(iterations: usize) -> Config {
        Config {
            schema: None,
            state_space: cobre_io::config::StateSpaceConfig::default(),
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                    limit: iterations as u32,
                }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    // ---------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------

    /// Backward-pass cut coefficient for an anticipated thermal with `lead_stages = 1`
    /// in a 2-stage system.
    ///
    /// One anticipated thermal (K=1, cost c_ant) and one regular thermal (cost c_reg) at a
    /// single bus; loads D_0, D_1; one one-hour block per stage; max_par_order = 0 so
    /// anticipated_slots_out.start = 0. The LP-builder divides every non-theta objective
    /// coefficient by COST_SCALE_FACTOR (call it K), so the stored cut lives in scaled units.
    ///
    /// Stage-1 LP (the anticipated decision column d_ant carries scaled cost c_reg/K):
    ///
    /// ```text
    ///   min  (c_reg/K) gt_reg + (c_reg/K) d_ant + theta
    ///   s.t. gt_reg + gt_ant = D_1            (load balance)
    ///        gt_ant - x_state = 0             (fishing, K=1)
    ///        x_state + d_ant = x_hat          (state-fixing, dual pi)
    ///        theta >= 0
    /// ```
    ///
    /// At the box optimum d_ant = 0, Q_scaled(x_hat) = (c_reg/K)(D_1 - x_hat), so the
    /// state-fixing dual is pi = -c_reg/K. With coefficients = dual (no sign flip), the
    /// coefficient is -c_reg/K and the intercept is Q_scaled(x_hat) - pi*x_hat = (c_reg/K)*D_1.
    #[test]
    fn two_stage_k1_anticipated_cut_coefficient_matches_analytical() {
        const K_MAX: usize = FIXTURE_K1.k_max;
        const D_1: f64 = FIXTURE_K1.loads_mw[1];
        const EXPECTED_COEFFICIENT: f64 = FIXTURE_K1.expected_coeff;
        const EXPECTED_INTERCEPT: f64 = C_REG * D_1 / COST_SCALE_FACTOR;

        let system = build_system(&FIXTURE_K1);
        let config = build_config(FIXTURE_K1.iterations);
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(
                &mut solver,
                &comm,
                FIXTURE_K1.iterations,
                ActiveSolver::new,
                None,
                None,
            )
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error must be None; got {:?}",
            outcome.error
        );

        let pool0 = &setup.fcf.pools[0];
        let active_count = pool0.active_count();
        assert_eq!(
            active_count, 1,
            "stage 0 FCF must contain exactly one active cut; got {active_count}",
        );

        let state = setup.stage_state();
        let ant_state_idx = state.anticipated_slots_out.start;
        assert_eq!(
            state.n_anticipated, 1,
            "fixture must have exactly one anticipated thermal",
        );
        assert_eq!(state.k_max, K_MAX, "fixture must have k_max = {K_MAX}");
        assert_eq!(
            ant_state_idx, 0,
            "with n_hydros=0 and max_par_order=0, anticipated_slots_out.start must be 0; got {ant_state_idx}",
        );

        let (slot, intercept, coefficients) = setup
            .fcf
            .active_cuts(0)
            .next()
            .expect("exactly one active cut must be retrievable from stage 0 pool");
        assert_eq!(
            coefficients.len(),
            state.anticipated_slots_out.end,
            "coefficient slice length must equal n_state",
        );

        let actual_coeff = coefficients[ant_state_idx];
        assert!(
            (actual_coeff - EXPECTED_COEFFICIENT).abs() < TOL,
            "cut coefficient at anticipated_slots_out index {ant_state_idx} \
         (slot={slot}, n_state={n_state}) does not match analytical value: \
         actual = {actual_coeff}, expected = {EXPECTED_COEFFICIENT} (= -c_reg/K = -{C_REG}/{COST_SCALE_FACTOR})",
            n_state = coefficients.len(),
        );

        assert!(
            (intercept - EXPECTED_INTERCEPT).abs() < TOL,
            "cut intercept does not match analytical value: actual = {intercept}, \
         expected = {EXPECTED_INTERCEPT} (= c_reg * D_1 / K = {C_REG} * {D_1} / {COST_SCALE_FACTOR})",
        );
    }

    /// Backward-pass cut-coefficient propagation for an anticipated thermal with
    /// `lead_stages = 2` in a 3-stage system.
    ///
    /// One anticipated thermal (K=2) and one regular thermal at a single bus; loads
    /// D_0, D_1, D_2; one one-hour block per stage; zero seeds; max_par_order = 0 so
    /// anticipated_slots_out.start = 0. Fishing rows are emitted at every stage in 0..n_stages.
    ///
    /// The stage-0 FCF cut is generated by backward t=0 (solving stage 1's LP), which carries
    /// the FCF cut produced earlier in the same sweep by backward t=1 (solving stage 2). Both
    /// stage-1 state-fixing-row duals equal -c_reg/COST_SCALE_FACTOR * BLOCK_HOURS: slot 0 is
    /// the same-stage fishing-equality dual (fishing is always active for every anticipated
    /// plant); slot 1 flows through the frozen stage-1 FCF cut, whose
    /// +c_reg/COST_SCALE_FACTOR * BLOCK_HOURS coefficient on x_state slot 1 originates from
    /// stage 2's slot-0 fishing dual, routed through the in-LP ring-shift definition row
    /// (`slot_0^out = slot_1^in`) that links stage 1's `anticipated_slots_out` to stage 2's
    /// `anticipated_state` incoming pin. So the stored stage-0 cut carries -0.0001 at both
    /// state slots.
    ///
    /// iterations = 5: backward t=0 consumes the cut just added to FCF[1] within the same
    /// iteration, so the propagated stage-2 subgradient reaches FCF[0] (the slot-1 cut) by
    /// iteration 1; the remaining iterations are margin and do not move the asserted cut.
    #[test]
    fn three_stage_k2_anticipated_cut_coefficient_propagates_correctly() {
        const K_MAX: usize = FIXTURE_K2.k_max;
        const N_ITERATIONS: usize = FIXTURE_K2.iterations;
        const EXPECTED_COEFF_SLOT0: f64 = FIXTURE_K2.expected_coeff;
        const EXPECTED_COEFF_SLOT1: f64 = FIXTURE_K2.expected_coeff;

        let system = build_system(&FIXTURE_K2);
        let config = build_config(FIXTURE_K2.iterations);
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(
                &mut solver,
                &comm,
                FIXTURE_K2.iterations,
                ActiveSolver::new,
                None,
                None,
            )
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error must be None; got {:?}",
            outcome.error,
        );

        let pool0 = &setup.fcf.pools[0];
        let active_count = pool0.active_count();
        assert!(
            active_count >= 1,
            "stage 0 FCF must contain at least one active cut after \
         {N_ITERATIONS} iterations; got {active_count}",
        );

        // Anticipated ring layout is `start + slot * n_anticipated + plant`; with
        // n_anticipated = 1, plant = 0 the slots are consecutive from `start`.
        let state = setup.stage_state();
        let ant_state_start = state.anticipated_slots_out.start;
        let slot0_idx = ant_state_start; // slot 0, plant 0
        let slot1_idx = ant_state_start + 1; // slot 1, plant 0
        assert_eq!(
            state.n_anticipated, 1,
            "fixture must have exactly one anticipated thermal",
        );
        assert_eq!(state.k_max, K_MAX, "fixture must have k_max = {K_MAX}");
        assert_eq!(
            ant_state_start, 0,
            "with n_hydros=0 and max_par_order=0, anticipated_slots_out.start must \
         be 0; got {ant_state_start}",
        );

        // The closed-form matches the iteration-1 cut only: it lands at slot 0 under
        // dense packing (iteration_base = 1, forward_passes = 1, per
        // CutPool::slot_index); later iterations add zero-subgradient cuts at shifted
        // trial points. Select slot 0 explicitly rather than the most-recent cut.
        let analytical = setup
            .fcf
            .active_cuts(0)
            .find(|(slot, _, _)| *slot == 0)
            .expect("iteration-1 cut (slot 0 under dense packing) must be present in stage 0 pool");
        let (slot, _intercept, coefficients) = analytical;

        assert_eq!(
            coefficients.len(),
            state.anticipated_slots_out.end,
            "coefficient slice length must equal n_state (= anticipated_slots_out.end \
         in this no-hydro fixture); got len={}, expected={}",
            coefficients.len(),
            state.anticipated_slots_out.end,
        );

        let actual_coeff_slot1 = coefficients[slot1_idx];
        assert!(
            (actual_coeff_slot1 - EXPECTED_COEFF_SLOT1).abs() < TOL,
            "stage 0 cut coefficient at anticipated_slots_out slot 1 \
         (state-vector index {slot1_idx}) does not match analytical value: \
         actual = {actual_coeff_slot1}, expected = {EXPECTED_COEFF_SLOT1} \
         (= -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS = \
         -{C_REG}/{COST_SCALE_FACTOR}*{BLOCK_HOURS}). \
         Source: dual flowing through the stage-1 FCF cut via the in-LP ring-shift \
         definition row (indexer.rs:state_to_lp_column). \
         Cut metadata: slot={slot}, n_state={n_state}, slot0_idx={slot0_idx}, \
         slot1_idx={slot1_idx}, iterations={N_ITERATIONS}",
            n_state = coefficients.len(),
        );

        let actual_coeff_slot0 = coefficients[slot0_idx];
        assert!(
            (actual_coeff_slot0 - EXPECTED_COEFF_SLOT0).abs() < TOL,
            "stage 0 cut coefficient at anticipated_slots_out slot 0 \
         (state-vector index {slot0_idx}) does not match analytical value: \
         actual = {actual_coeff_slot0}, expected = {EXPECTED_COEFF_SLOT0} \
         (= -C_REG / COST_SCALE_FACTOR * BLOCK_HOURS = \
         -{C_REG}/{COST_SCALE_FACTOR}*{BLOCK_HOURS}). \
         Source: same-stage fishing equality dual at stage 1; the fishing \
         constraint is always active for every anticipated plant. \
         Cut metadata: slot={slot}, n_state={n_state}, slot0_idx={slot0_idx}, \
         slot1_idx={slot1_idx}, iterations={N_ITERATIONS}",
            n_state = coefficients.len(),
        );
    }

    /// Backward-pass cut-coefficient propagation for an anticipated thermal with
    /// `lead_stages = 3` in a 4-stage system.
    ///
    /// One anticipated thermal (K=3, cost $50/MWh, max 50 MW), one regular thermal
    /// (cost $100/MWh, max 100 MW), loads 5, 10, 15, 30 MW, zero seeds, one-hour blocks.
    /// Fishing rows are emitted at every stage in 0..n_stages. All three stage-0 slots receive
    /// -c_reg/COST_SCALE_FACTOR via distinct paths:
    /// - slot 0: direct fishing dual at stage 1 (solving stage 2);
    /// - slot 1: stage-2 fishing dual via one in-LP ring-shift definition row through
    ///   stage-1's FCF cut;
    /// - slot 2: stage-3 fishing dual via two successive ring-shift definition rows
    ///   (stage-2 then stage-1 FCF cuts), reaching slot 2 at stage 0.
    ///
    /// See `StateSpace::state_to_lp_column` for the full algebraic chain.
    #[test]
    fn four_stage_k3_anticipated_cut_coefficient_propagates_correctly() {
        const K_MAX: usize = FIXTURE_K3.k_max;
        const N_ITERATIONS: usize = FIXTURE_K3.iterations;
        const EXPECTED_COEFF_SLOT0: f64 = FIXTURE_K3.expected_coeff;
        const EXPECTED_COEFF_SLOT1: f64 = FIXTURE_K3.expected_coeff;
        const EXPECTED_COEFF_SLOT2: f64 = FIXTURE_K3.expected_coeff;

        let system = build_system(&FIXTURE_K3);
        let config = build_config(FIXTURE_K3.iterations);
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(
                &mut solver,
                &comm,
                FIXTURE_K3.iterations,
                ActiveSolver::new,
                None,
                None,
            )
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error must be None; got {:?}",
            outcome.error,
        );

        let pool0 = &setup.fcf.pools[0];
        let active_count = pool0.active_count();
        assert!(
            active_count >= 1,
            "stage 0 FCF must contain at least one active cut after \
         {N_ITERATIONS} iterations; got {active_count}",
        );

        // Anticipated ring layout is `start + slot * n_anticipated + plant`; with
        // n_anticipated = 1, plant = 0 the slots are consecutive from `start`.
        let state = setup.stage_state();
        let ant_state_start = state.anticipated_slots_out.start;
        let slot0_idx = ant_state_start;
        let slot1_idx = ant_state_start + 1;
        let slot2_idx = ant_state_start + 2;
        assert_eq!(
            state.n_anticipated, 1,
            "fixture must have exactly one anticipated thermal",
        );
        assert_eq!(state.k_max, K_MAX, "fixture must have k_max = {K_MAX}");
        assert_eq!(
            ant_state_start, 0,
            "with n_hydros=0 and max_par_order=0, anticipated_slots_out.start must \
         be 0; got {ant_state_start}",
        );

        // The analytical match is the iteration-1 cut (slot 0 under dense packing,
        // per CutPool::slot_index): its three-stage propagation chain completes at
        // backward t=0. Later iterations add cuts at trial points with a different
        // active basis.
        let analytical = setup
            .fcf
            .active_cuts(0)
            .find(|(slot, _, _)| *slot == 0)
            .expect("iteration-1 cut (slot 0 under dense packing) must be present in stage 0 pool");
        let (_slot, _intercept, coefficients) = analytical;

        assert_eq!(
            coefficients.len(),
            state.anticipated_slots_out.end,
            "coefficient slice length must equal n_state (= anticipated_slots_out.end \
         in this no-hydro fixture); got len={}, expected={}",
            coefficients.len(),
            state.anticipated_slots_out.end,
        );

        let actual_coeff_slot2 = coefficients[slot2_idx];
        assert!(
            (actual_coeff_slot2 - EXPECTED_COEFF_SLOT2).abs() < TOL,
            "slot 2 coefficient {actual_coeff_slot2} != {EXPECTED_COEFF_SLOT2} \
         (stage-3 fishing dual via two FCF frozen cuts and successive in-LP \
         ring-shift definition rows)",
        );

        let actual_coeff_slot1 = coefficients[slot1_idx];
        assert!(
            (actual_coeff_slot1 - EXPECTED_COEFF_SLOT1).abs() < TOL,
            "slot 1 coefficient {actual_coeff_slot1} != {EXPECTED_COEFF_SLOT1} \
         (stage-2 fishing dual via one in-LP ring-shift definition row through \
         stage-1 FCF cut)",
        );

        let actual_coeff_slot0 = coefficients[slot0_idx];
        assert!(
            (actual_coeff_slot0 - EXPECTED_COEFF_SLOT0).abs() < TOL,
            "slot 0 coefficient {actual_coeff_slot0} != {EXPECTED_COEFF_SLOT0} \
         (stage-1 fishing equality dual under always-active predicate)",
        );
    }
}
mod anticipated_pre_horizon_seed_delivery {
    //! Pre-horizon seed-delivery integration tests for an anticipated thermal across
    //! lead_stages K = 1, 2, 3. Each test trains a small in-code study, runs a
    //! one-scenario simulation, and asserts that the matured ring-buffer seeds are
    //! delivered at the early stages, that anticipated decisions saturate within
    //! bounds, that the ring-buffer shift maps committed_at(t) ≈ decision_at(t−K),
    //! and that the observed cost stays under a per-K analytical upper bound. Each
    //! K's derivation and cost bound live on its test function.

    use cobre_core::entities::{
        bus::DeficitSegment,
        hydro::{HydroGenerationModel, HydroPenalties},
        thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, PolicyGraph, PolicyGraphType, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroStageBounds, HydroStagePenalties,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_io::config::{SimulationSelection, TrainingSelection};

    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };
    use super::common::run_simulation;

    // ---------------------------------------------------------------------------
    // Per-K fixture table
    // ---------------------------------------------------------------------------

    /// Per-K parameters for the pre-horizon seed-delivery fixtures. Each `#[test]`
    /// builds an independent `System` from one entry; the entity IDs sit in disjoint
    /// decades (30s / 40s / 50s), themselves disjoint from the 2..7 range the sibling
    /// anticipated tests use, so a combined nextest run attributes failures
    /// unambiguously per entity.
    struct SeedDeliveryFixture {
        n_stages: usize,
        /// Anticipated `lead_stages` K — also the ring-buffer depth `k_max` and the
        /// `kN` suffix in each entity name.
        k: usize,
        bus_id: EntityId,
        hydro_id: EntityId,
        anticipated_id: EntityId,
        backup_id: EntityId,
        /// Anticipated ring-buffer seeds, MW (length `k`).
        seeds_mw: &'static [f64],
        iterations: usize,
    }

    const FIXTURE_K1: SeedDeliveryFixture = SeedDeliveryFixture {
        n_stages: 5,
        k: 1,
        bus_id: EntityId(1),
        hydro_id: EntityId(30),
        anticipated_id: EntityId(31),
        backup_id: EntityId(32),
        seeds_mw: &[100.0],
        iterations: 1,
    };

    const FIXTURE_K2: SeedDeliveryFixture = SeedDeliveryFixture {
        n_stages: 5,
        k: 2,
        bus_id: EntityId(1),
        hydro_id: EntityId(41),
        anticipated_id: EntityId(42),
        backup_id: EntityId(43),
        seeds_mw: &[80.0, 50.0],
        iterations: 5,
    };

    const FIXTURE_K3: SeedDeliveryFixture = SeedDeliveryFixture {
        n_stages: 6,
        k: 3,
        bus_id: EntityId(1),
        hydro_id: EntityId(51),
        anticipated_id: EntityId(52),
        backup_id: EntityId(53),
        seeds_mw: &[50.0, 30.0, 10.0],
        iterations: 5,
    };

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// Build the `System` for one seed-delivery fixture: one anticipated thermal at
    /// `fixture.anticipated_id`, one backup at `fixture.backup_id`, a trivial hydro at
    /// `fixture.hydro_id`, load 150 MW, ring-buffer seed `fixture.seeds_mw`.
    ///
    /// Constructing `System` directly via `SystemBuilder` bypasses the `cobre-io`
    /// validator that rejects non-zero `values_mw`: the non-zero seed is the
    /// deliberate fixture; the rejection rule applies only to JSON input through
    /// `load_case`. The $10/MWh anticipated vs $5000/MWh backup asymmetry saturates
    /// anticipated dispatch at max_gen, and `annual_discount_rate = 0.0` collapses
    /// every discount factor to 1.0 so each test's analytical cost derivation is exact.
    fn build_system(fixture: &SeedDeliveryFixture) -> cobre_core::System {
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

        let thermal_ant = make_thermal(
            fixture.anticipated_id,
            ThermalSpec {
                name: format!("T_ant_seed_k{k}"),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: fixture.bus_id,
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
                cost_per_mwh: 10.0,
                anticipated_config: Some(AnticipatedConfig::LeadStages(k as u32)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            fixture.backup_id,
            ThermalSpec {
                name: format!("T_backup_seed_k{k}"),
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

        // Zero inflow and 1 MW max_gen keep the system firmly in the thermal regime.
        let hydro = make_hydro(
            fixture.hydro_id,
            HydroSpec {
                name: format!("H1_seed_k{k}"),
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
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 1.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 1.0,
                max_diversion_m3s: None,
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

        // The padding region [n_stages, n_stages + k) is the delivery-stage axis read
        // by `fill_anticipated_columns`; it must carry per-thermal costs so the
        // decision column's objective coefficient is non-zero.
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
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 200.0,
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
        );
        // Thermal index 0 = anticipated (cheap); index 1 = backup (expensive). Without
        // these per-thermal cost overrides the LP has no incentive to commit
        // anticipated capacity and decision_at(t) collapses to zero, masking the
        // regression.
        for s in 0..thermal_axis {
            bounds.thermal_bounds_mut(0, s).cost_per_mwh = 10.0;
            bounds.thermal_block_base_mut(0, s).max_generation_mw = 200.0;
            bounds.thermal_bounds_mut(1, s).cost_per_mwh = 5000.0;
            bounds.thermal_block_base_mut(1, s).max_generation_mw = 500.0;
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

        // Seed the anticipated ring buffer; distinct seed values catch slot-swap bugs
        // that identical values would mask across the pre-horizon shifts.
        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: fixture.hydro_id,
                value_hm3: 0.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: fixture.anticipated_id,
                values_mw: fixture.seeds_mw.to_vec(),
            }],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        let policy_graph = PolicyGraph {
            stage_discount_rate_overrides: std::collections::HashMap::new(),
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: vec![],
            nodes: Vec::new(),
            season_map: None,
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
            .policy_graph(policy_graph)
            .build()
            .expect("build_system: valid system")
    }

    // ---------------------------------------------------------------------------
    // Config builder
    // ---------------------------------------------------------------------------

    fn build_config(iterations: usize) -> Config {
        Config {
            schema: None,
            state_space: cobre_io::config::StateSpaceConfig::default(),
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                    limit: iterations as u32,
                }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig {
                enabled: true,
                io_channel_capacity: 8,
                selection: Some(SimulationSelection::Sampled { num_scenarios: 1 }),
                ..IoSimulationConfig::default()
            },
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    // ---------------------------------------------------------------------------
    // Tests
    // ---------------------------------------------------------------------------

    /// Pre-horizon seed delivery at stage 0 with K=1 and the always-active fishing
    /// predicate. The always-active fishing equality at stage 0 pins the anticipated
    /// thermal to slot 0 of the ring buffer (the 100 MW seed) and the cost-zeroing
    /// predicate accepts that delivery at zero LP cost.
    ///
    /// Verifies that the 100 MW seed is delivered at stage 0 and that the ring-buffer
    /// shift propagates in-study decisions for stages 1–4 (committed seed at stage 0,
    /// decision saturation, ring-buffer shift, cost upper bound — derived inline at
    /// each AC block below).
    ///
    /// Cost bound: stage-0 backup carries 150 − 100 = 50 MW × $5000/MWh × 744 h =
    /// $186,000,000 (the 100 MW seed delivers at zero LP cost); the active-decision
    /// ceiling is 4 decision stages × 200 MW × $10/MWh × 744 h = $5,952,000 (the LP
    /// may commit less if cuts are loose, never more); plus a $1,000 tolerance.
    #[test]
    fn pre_horizon_seed_delivers_at_stage_zero_k1() {
        const STAGE_0_BACKUP_COST_USD: f64 = (150.0 - 100.0) * 744.0 * 5000.0;
        const MAX_DECISION_COST_USD: f64 = 4.0 * 200.0 * 744.0 * 10.0;
        const COST_TOLERANCE_USD: f64 = 1_000.0;
        const EXPECTED_TOTAL_UPPER_BOUND_USD: f64 =
            STAGE_0_BACKUP_COST_USD + MAX_DECISION_COST_USD + COST_TOLERANCE_USD;

        let system = build_system(&FIXTURE_K1);
        let config = build_config(FIXTURE_K1.iterations);
        let mut setup = build_setup_in_code(system, &config);

        // One iteration suffices: the fishing equality pins the seed regardless of cut
        // quality; the generous cost bound absorbs a loose 1-iteration cut.
        let scenario_results = run_simulation(&mut setup, FIXTURE_K1.iterations);

        assert_eq!(
            scenario_results.len(),
            1,
            "simulation must stream exactly one scenario result",
        );
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            5,
            "scenario must contain one record per study stage (n_stages=5)",
        );

        let anticipated_thermal_id: i32 = 31;
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

        let c0 = committed_at(0).expect(
            "committed_at(0) is None. \
         The always-active fishing predicate is not delivering the 100 MW seed \
         at stage 0. Legacy behaviour: predicate `K_i > stage_idx` gated \
         fishing off at stage 0 with K=1.",
        );
        assert!(
            (c0 - 100.0).abs() < 1e-6,
            "committed_at(0) = {c0} MW, expected 100.0 MW (the seed). \
         The fishing equality at stage 0 must pin the anticipated thermal to \
         slot 0 = 100.0 MW of the ring buffer.",
        );

        // Active-decision stages are t ∈ {0,1,2,3} (t + K < n_stages, i.e. t + 1 < 5).
        for t in 0..4_usize {
            let dt = decision_at(t).unwrap_or_else(|| {
                panic!(
                    "decision_at({t}) is None; anticipated thermal id=31 \
                 was not found in stage {t} thermals or anticipated_decision_mw \
                 is absent (stage {t} is an active-decision stage: {t} + 1 < 5)",
                )
            });
            assert!(
                dt.abs() > 1e-6,
                "decision_at({t}) = {dt} MW is zero (≤ 1e-6). \
             The LP should commit a non-trivial anticipated amount at stage {t} \
             to avoid $5000/MWh backup at the delivery stage.",
            );
            assert!(
                dt <= 200.0 + 1e-6,
                "decision_at({t}) = {dt} MW exceeds max_gen=200 MW. \
             This indicates a bounds violation in the LP.",
            );
        }

        // Ring-buffer shift invariant: after the shift at the end of stage t-1, slot 0
        // holds the in-study decision from t-1, which stage t's fishing equality pins.
        for t in 1..5_usize {
            let ct = committed_at(t).unwrap_or_else(|| {
                panic!(
                    "committed_at({t}) is None; expected a matured \
                 commitment from decision at stage {}",
                    t - 1,
                )
            });
            let d_prev = decision_at(t - 1).unwrap_or_else(|| {
                panic!(
                    "decision_at({}) is None (needed to check ring-buffer \
                 invariant at stage {t})",
                    t - 1,
                )
            });
            assert!(
                (ct - d_prev).abs() < 1e-6,
                "ring-buffer shift: committed_at({t}) = {ct} MW should \
             equal decision_at({}) = {d_prev} MW (within 1e-6 MW). \
             The ring buffer is not correctly propagating in-study decisions.",
                t - 1,
            );
        }

        // Sum immediate_cost (LP objective minus theta), not total_cost — total_cost
        // includes the theta approximation artefact.
        let observed_total: f64 = scenario
            .stages
            .iter()
            .flat_map(|st| st.costs.iter().map(|c| c.immediate_cost))
            .sum();

        assert!(
            observed_total <= EXPECTED_TOTAL_UPPER_BOUND_USD,
            "observed_total = ${observed_total:.2} exceeds upper bound \
         ${EXPECTED_TOTAL_UPPER_BOUND_USD:.2}. \
         Breakdown: STAGE_0_BACKUP_COST_USD=${STAGE_0_BACKUP_COST_USD:.2}, \
         MAX_DECISION_COST_USD=${MAX_DECISION_COST_USD:.2}, \
         COST_TOLERANCE_USD=${COST_TOLERANCE_USD:.2}. \
         If the seed is not delivered (legacy predicate), stage-0 backup covers \
         150 MW instead of 50 MW, producing ~$558M >> $191.95M.",
        );
    }

    /// Pre-horizon seed delivery across two pre-horizon stages with K=2 and the
    /// always-active fishing predicate.
    ///
    /// With `K = 2`, `n_stages = 5`, a single anticipated thermal (id=42), and
    /// `past_anticipated_commitments.values_mw = [80.0, 50.0]`, the LP must:
    ///
    /// 1. Deliver `committed_at(0) == 80.0 MW` — the always-active fishing
    ///    equality at stage 0 pins the anticipated thermal to slot 0 of the ring
    ///    buffer, which holds the 80.0 MW seed (`values_mw[0]`). The cost-zeroing
    ///    predicate zeros the per-block objective for this column so the LP
    ///    accepts the delivery at zero additional cost.
    ///
    /// 2. Deliver `committed_at(1) == 50.0 MW` — the in-LP ring-shift definition
    ///    row moves slot 1 (`values_mw[1] = 50.0`) into slot 0 at the start of
    ///    stage 1. Stage 1's always-active fishing equality then reads slot 0 =
    ///    50.0 MW. This is the K=2-specific assertion that the K=1 delivery test
    ///    cannot reach: K=1 has only one pre-horizon stage, so there is no
    ///    ring-buffer shift between two pre-horizon stages to exercise.
    ///
    /// 3. Satisfy `committed_at(t) ≈ decision_at(t-2)` for t ∈ {2,3,4} — the
    ///    K=2 ring-buffer matures decisions two stages after they are made. With
    ///    K=2, the decision written at stage t occupies slot `K-1 = 1` in the
    ///    outgoing state, which shifts into slot 0 after two forward steps, at
    ///    which point the fishing equality delivers it. This is the t-2 offset
    ///    (compare: K=1 delivery test uses t-1 offset).
    ///
    /// 4. Saturate `decision_at(t) ≈ 200 MW` (max_gen) for t ∈ {0,1,2} (stages
    ///    where `t + K < n_stages`, i.e., `t + 2 < 5`) — the anticipated thermal
    ///    costs $10/MWh vs the backup's $5000/MWh, and the per-block cost of the
    ///    decision column is non-zero only at the decision stage, so the LP
    ///    saturates commitment to avoid future backup dispatch.
    ///
    /// 5. Satisfy the analytical cost bound:
    ///    - Stage 0: seed delivers 80 MW; backup covers 70 MW
    ///      × $5000/MWh × 744 h = $260,400,000.
    ///    - Stage 1: shifted seed delivers 50 MW; backup covers 100 MW
    ///      × $5000/MWh × 744 h = $372,000,000.
    ///    - Stages 2–4 delivery: anticipated covers ≥ 150 MW load (zeroed cost).
    ///    - Decision cost ≤ 3 × 200 MW × $10/MWh × 744 h = $4,464,000.
    ///    - Total ≤ $636,865,000.
    #[test]
    fn pre_horizon_seed_delivers_pre_horizon_stages_k2() {
        const STAGE_0_BACKUP_COST_USD: f64 = (150.0 - 80.0) * 744.0 * 5000.0;
        const STAGE_1_BACKUP_COST_USD: f64 = (150.0 - 50.0) * 744.0 * 5000.0;
        const MAX_DECISION_COST_USD: f64 = 3.0 * 200.0 * 744.0 * 10.0;
        const COST_TOLERANCE_USD: f64 = 1_000.0;
        const EXPECTED_TOTAL_UPPER_BOUND_USD: f64 = STAGE_0_BACKUP_COST_USD
            + STAGE_1_BACKUP_COST_USD
            + MAX_DECISION_COST_USD
            + COST_TOLERANCE_USD;

        let system = build_system(&FIXTURE_K2);
        let config = build_config(FIXTURE_K2.iterations);
        let mut setup = build_setup_in_code(system, &config);

        // 5 iterations: after 1, stage-1 decisions are too loose to satisfy the cost
        // bound.
        let scenario_results = run_simulation(&mut setup, FIXTURE_K2.iterations);

        assert_eq!(
            scenario_results.len(),
            1,
            "simulation must stream exactly one scenario result",
        );
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            5,
            "scenario must contain one record per study stage (n_stages=5)",
        );

        let anticipated_thermal_id: i32 = 42;
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

        let c0 = committed_at(0).expect(
            "committed_at(0) is None. \
         The always-active fishing predicate is not delivering the 80 MW seed \
         at stage 0. Legacy behaviour: predicate `K_i > stage_idx` gated \
         fishing off at pre-horizon stages.",
        );
        assert!(
            (c0 - 80.0).abs() < 1e-6,
            "committed_at(0) = {c0} MW, expected 80.0 MW (values_mw[0]). \
         The fishing equality at stage 0 must pin the anticipated thermal to \
         slot 0 = 80.0 MW of the ring buffer.",
        );

        // K=2-specific: ring-buffer shift between pre-horizon stages 0→1.
        let c1 = committed_at(1).expect(
        "committed_at(1) is None. \
         The fishing constraint is always active for every anticipated plant, so it must be active at stage 1. \
         If committed_at(1) is None, the fishing constraint is absent for stage 1.",
    );
        assert!(
            (c1 - 50.0).abs() < 1e-6,
            "committed_at(1) = {c1} MW, expected 50.0 MW (values_mw[1]). \
         The in-LP ring-shift definition row must move slot 1 (50.0 MW) into \
         slot 0 at the start of stage 1, and the fishing equality must read \
         that value. If the result is 80.0 MW, the ring-buffer shift is not \
         moving slot 1 into slot 0 between pre-horizon stages.",
        );

        // K=2 ring-buffer: decisions mature 2 stages after being made.
        for t in 2..5_usize {
            let ct = committed_at(t).unwrap_or_else(|| {
                panic!(
                    "committed_at({t}) is None; expected a matured \
                 commitment from decision at stage {}",
                    t - 2,
                )
            });
            let d_prev2 = decision_at(t - 2).unwrap_or_else(|| {
                panic!(
                    "decision_at({}) is None (needed to check K=2 \
                 ring-buffer invariant at stage {t})",
                    t - 2,
                )
            });
            assert!(
                (ct - d_prev2).abs() < 1e-6,
                "K=2 ring-buffer shift: committed_at({t}) = {ct} MW should \
             equal decision_at({}) = {d_prev2} MW (within 1e-6 MW). \
             With K=2, decisions mature two stages later (slot K-1=1 shifts into \
             slot 0 after two forward steps). The ring buffer is not correctly \
             propagating in-study decisions.",
                t - 2,
            );
        }

        // Active-decision stages: t + 2 < 5; LP saturates on cost ratio.
        for t in 0..3_usize {
            let dt = decision_at(t).unwrap_or_else(|| {
                panic!(
                    "decision_at({t}) is None; anticipated thermal id=42 \
                 was not found in stage {t} thermals or anticipated_decision_mw \
                 is absent (stage {t} is an active-decision stage: {t} + 2 < 5)",
                )
            });
            assert!(
                dt.abs() > 1e-6,
                "decision_at({t}) = {dt} MW is zero (≤ 1e-6). \
             The LP should commit a non-trivial anticipated amount at stage {t} \
             to avoid $5000/MWh backup at the delivery stage (stage {}).",
                t + 2,
            );
            assert!(
                dt <= 200.0 + 1e-6,
                "decision_at({t}) = {dt} MW exceeds max_gen=200 MW. \
             This indicates a bounds violation in the LP.",
            );
        }

        // Use immediate_cost, not total_cost (which includes the theta approximation
        // artefact).
        let observed_total: f64 = scenario
            .stages
            .iter()
            .flat_map(|st| st.costs.iter().map(|c| c.immediate_cost))
            .sum();

        assert!(
            observed_total <= EXPECTED_TOTAL_UPPER_BOUND_USD,
            "observed_total = ${observed_total:.2} exceeds upper bound \
         ${EXPECTED_TOTAL_UPPER_BOUND_USD:.2}. \
         Breakdown: STAGE_0_BACKUP_COST_USD=${STAGE_0_BACKUP_COST_USD:.2}, \
         STAGE_1_BACKUP_COST_USD=${STAGE_1_BACKUP_COST_USD:.2}, \
         MAX_DECISION_COST_USD=${MAX_DECISION_COST_USD:.2}, \
         COST_TOLERANCE_USD=${COST_TOLERANCE_USD:.2}. \
         If neither seed is delivered, stage-0+1 backup covers 300 MW total \
         producing ~$1,116M >> $636.865M.",
        );
    }

    /// Pre-horizon seed delivery across three pre-horizon stages with K=3 and the
    /// always-active fishing predicate.
    ///
    /// With `K = 3`, `n_stages = 6`, a single anticipated thermal (id=52), and
    /// `past_anticipated_commitments.values_mw = [50.0, 30.0, 10.0]`, the LP must:
    ///
    /// 1. Deliver `committed_at(0) == 50.0 MW` — the always-active fishing
    ///    equality at stage 0 pins the anticipated thermal to slot 0 of the ring
    ///    buffer, which holds the 50.0 MW seed (`values_mw[0]`). The cost-zeroing
    ///    predicate zeros the per-block objective for this column so the LP
    ///    accepts the delivery at zero additional cost.
    ///
    /// 2. Deliver `committed_at(1) == 30.0 MW` — the in-LP ring-shift definition
    ///    row moves slot 1 (`values_mw[1] = 30.0`) into slot 0 at
    ///    the start of stage 1. Stage 1's always-active fishing equality then
    ///    reads slot 0 = 30.0 MW. This is one of the two K=3-specific assertions
    ///    that the K=1 and K=2 delivery tests cannot reach: K=3 has three
    ///    pre-horizon stages, with two ring-buffer shifts between them.
    ///
    /// 3. Deliver `committed_at(2) == 10.0 MW` — after two ring-buffer shifts,
    ///    slot 0 holds `values_mw[2] = 10.0`. Stage 2's always-active fishing
    ///    equality delivers it at zero LP cost. This is the deepest pre-horizon
    ///    delivery assertion in the entire anticipated test suite.
    ///
    /// 4. Satisfy `committed_at(t) ≈ decision_at(t-3)` for t ∈ {3, 4, 5} — the
    ///    K=3 ring-buffer matures decisions three stages after they are committed.
    ///    With K=3, the decision written at stage t occupies slot `K-1 = 2` in
    ///    the outgoing state, which shifts into slot 0 after three forward steps,
    ///    at which point the fishing equality delivers it. This is the t-3 offset
    ///    (compare: K=1 uses t-1, K=2 uses t-2).
    ///
    /// 5. Saturate `decision_at(t) > 0` and `≤ max_gen + 1e-6` for t ∈ {0, 1, 2}
    ///    (stages where `t + K < n_stages`, i.e., `t + 3 < 6`, giving t ∈ {0,1,2})
    ///    — the anticipated thermal costs $10/MWh vs the backup's $5000/MWh, and
    ///    the per-block cost of the decision column is non-zero only at the
    ///    decision stage, so the LP commits a non-trivial amount to avoid future
    ///    backup dispatch.
    ///
    /// 6. Satisfy the analytical cost bound:
    ///    - Stage 0: seed delivers 50 MW; backup covers 100 MW
    ///      × $5000/MWh × 744 h = $372,000,000.
    ///    - Stage 1: shifted seed delivers 30 MW; backup covers 120 MW
    ///      × $5000/MWh × 744 h = $446,400,000.
    ///    - Stage 2: doubly-shifted seed delivers 10 MW; backup covers 140 MW
    ///      × $5000/MWh × 744 h = $520,800,000.
    ///    - Stages 3–5 delivery: anticipated delivers ≥ 150 MW load (zeroed cost).
    ///    - Decision cost ≤ 3 × 200 MW × $10/MWh × 744 h = $4,464,000.
    ///    - Tolerance: $1,000.
    ///    - Total upper bound: $1,343,665,000.
    #[test]
    fn pre_horizon_seed_delivers_three_pre_horizon_stages_k3() {
        const STAGE_0_BACKUP_COST_USD: f64 = (150.0 - 50.0) * 744.0 * 5000.0;
        const STAGE_1_BACKUP_COST_USD: f64 = (150.0 - 30.0) * 744.0 * 5000.0;
        const STAGE_2_BACKUP_COST_USD: f64 = (150.0 - 10.0) * 744.0 * 5000.0;
        const MAX_DECISION_COST_USD: f64 = 3.0 * 200.0 * 744.0 * 10.0;
        const COST_TOLERANCE_USD: f64 = 1_000.0;
        const EXPECTED_TOTAL_UPPER_BOUND_USD: f64 = STAGE_0_BACKUP_COST_USD
            + STAGE_1_BACKUP_COST_USD
            + STAGE_2_BACKUP_COST_USD
            + MAX_DECISION_COST_USD
            + COST_TOLERANCE_USD;

        let system = build_system(&FIXTURE_K3);
        let config = build_config(FIXTURE_K3.iterations);
        let mut setup = build_setup_in_code(system, &config);

        // 5 iterations let cuts sharpen so stage-2 decisions reach max_gen, covering
        // stage-5 delivery at zero backup cost.
        let scenario_results = run_simulation(&mut setup, FIXTURE_K3.iterations);

        assert_eq!(
            scenario_results.len(),
            1,
            "simulation must stream exactly one scenario result",
        );
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            6,
            "scenario must contain one record per study stage (n_stages=6)",
        );

        let anticipated_thermal_id: i32 = 52;
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

        let c0 = committed_at(0).expect(
            "committed_at(0) is None. \
         The always-active fishing predicate is not delivering the 50 MW seed \
         at stage 0. Legacy behaviour: predicate `K_i > stage_idx` gated \
         fishing off at pre-horizon stages.",
        );
        assert!(
            (c0 - 50.0).abs() < 1e-6,
            "committed_at(0) = {c0} MW, expected 50.0 MW (values_mw[0]). \
         The fishing equality at stage 0 must pin the anticipated thermal to \
         slot 0 = 50.0 MW of the ring buffer.",
        );

        let c1 = committed_at(1).expect(
        "committed_at(1) is None. \
         The fishing constraint is always active for every anticipated plant, so it must be active at stage 1. \
         If committed_at(1) is None, the fishing constraint is absent for stage 1.",
    );
        assert!(
            (c1 - 30.0).abs() < 1e-6,
            "committed_at(1) = {c1} MW, expected 30.0 MW (values_mw[1]). \
         The in-LP ring-shift definition row must move slot 1 (30.0 MW) into \
         slot 0 at the start of stage 1, and the fishing equality must read \
         that value. If the result is 50.0 MW, the first ring-buffer shift is not \
         moving slot 1 into slot 0 between pre-horizon stages 0 and 1.",
        );

        let c2 = committed_at(2).expect(
        "committed_at(2) is None. \
         The fishing constraint is always active for every anticipated plant, so it must be active at stage 2. \
         If committed_at(2) is None, the fishing constraint is absent for stage 2.",
    );
        assert!(
            (c2 - 10.0).abs() < 1e-6,
            "committed_at(2) = {c2} MW, expected 10.0 MW (values_mw[2]). \
         After two ring-buffer shifts, slot 0 must hold 10.0 MW. \
         If the result is 30.0 MW, the second ring-buffer shift (between stages 1 \
         and 2) is not moving slot 1 (10.0 MW) into slot 0 correctly. \
         If the result is 50.0 MW, neither shift has occurred.",
        );

        for t in 3..6_usize {
            let ct = committed_at(t).unwrap_or_else(|| {
                panic!(
                    "committed_at({t}) is None; expected a matured \
                 commitment from decision at stage {}",
                    t - 3,
                )
            });
            let d_prev3 = decision_at(t - 3).unwrap_or_else(|| {
                panic!(
                    "decision_at({}) is None (needed to check K=3 \
                 ring-buffer invariant at stage {t})",
                    t - 3,
                )
            });
            assert!(
                (ct - d_prev3).abs() < 1e-6,
                "K=3 ring-buffer shift: committed_at({t}) = {ct} MW should \
             equal decision_at({}) = {d_prev3} MW (within 1e-6 MW). \
             With K=3, decisions mature three stages later (slot K-1=2 shifts into \
             slot 0 after three forward steps). The ring buffer is not correctly \
             propagating in-study decisions.",
                t - 3,
            );
        }

        for t in 0..3_usize {
            let dt = decision_at(t).unwrap_or_else(|| {
                panic!(
                    "decision_at({t}) is None; anticipated thermal id=52 \
                 was not found in stage {t} thermals or anticipated_decision_mw \
                 is absent (stage {t} is an active-decision stage: {t} + 3 < 6)",
                )
            });
            assert!(
                dt.abs() > 1e-6,
                "decision_at({t}) = {dt} MW is zero (≤ 1e-6). \
             The LP should commit a non-trivial anticipated amount at stage {t} \
             to avoid $5000/MWh backup at the delivery stage (stage {}).",
                t + 3,
            );
            assert!(
                dt <= 200.0 + 1e-6,
                "decision_at({t}) = {dt} MW exceeds max_gen=200 MW. \
             This indicates a bounds violation in the LP.",
            );
        }

        // Sum immediate_cost, not total_cost — total_cost includes the theta
        // approximation artefact.
        let observed_total: f64 = scenario
            .stages
            .iter()
            .flat_map(|st| st.costs.iter().map(|c| c.immediate_cost))
            .sum();

        assert!(
            observed_total <= EXPECTED_TOTAL_UPPER_BOUND_USD,
            "observed_total = ${observed_total:.2} exceeds upper bound \
         ${EXPECTED_TOTAL_UPPER_BOUND_USD:.2}. \
         Breakdown: STAGE_0_BACKUP_COST_USD=${STAGE_0_BACKUP_COST_USD:.2}, \
         STAGE_1_BACKUP_COST_USD=${STAGE_1_BACKUP_COST_USD:.2}, \
         STAGE_2_BACKUP_COST_USD=${STAGE_2_BACKUP_COST_USD:.2}, \
         MAX_DECISION_COST_USD=${MAX_DECISION_COST_USD:.2}, \
         COST_TOLERANCE_USD=${COST_TOLERANCE_USD:.2}. \
         If none of the seeds are delivered, 3 pre-horizon stages use backup \
         for all 150 MW: 3 × 150 MW × 744 h × $5000/MWh = $1,674,000,000 >> \
         $1,343,665,000.",
        );
    }
}
mod anticipated_d_t_saturation {
    //! `d_t`-saturation regression tests for an anticipated thermal across lead_stages
    //! K = 2, 3. Each test trains a small in-code study, runs a one-scenario
    //! simulation, and asserts that the anticipated-decision variable `d_t` commits to
    //! the load level at every active stage (`t + K < n_stages`) and is absent at the
    //! strict-boundary inactive stages. The K=3 test additionally checks the
    //! ring-buffer shift `committed_at(t) ≈ decision_at(t − K)`. Each K's
    //! economic-reasoning derivation lives on its test function.

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
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroStageBounds, HydroStagePenalties,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_io::config::{SimulationSelection, TrainingSelection};

    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };
    use super::common::run_simulation;

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
                anticipated_config: Some(AnticipatedConfig::LeadStages(k as u32)),
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
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 1.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 1.0,
                max_diversion_m3s: None,
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
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 200.0,
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
        );
        for s in 0..thermal_axis {
            bounds.thermal_bounds_mut(0, s).cost_per_mwh = 10.0; // index 0 = anticipated (cheap)
            bounds.thermal_block_base_mut(0, s).max_generation_mw = 200.0;
            bounds.thermal_bounds_mut(1, s).cost_per_mwh = 5000.0; // index 1 = backup (expensive)
            bounds.thermal_block_base_mut(1, s).max_generation_mw = 500.0;
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
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: anticipated_id,
                values_mw: fixture.seeds_mw.to_vec(),
            }],
            recent_observations: vec![],
            past_defluences: vec![],
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
            state_space: cobre_io::config::StateSpaceConfig::default(),
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                    limit: iterations as u32,
                }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig {
                enabled: true,
                io_channel_capacity: 8,
                selection: Some(SimulationSelection::Sampled { num_scenarios: 1 }),
                ..IoSimulationConfig::default()
            },
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

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

        let load_mw = 150.0_f64;
        let tol = 1e-3_f64;
        for t in &active_stages {
            let d_t = decision_at(*t).unwrap_or_else(|| {
                panic!(
                    "anticipated_decision_mw must be Some at active stage t={t} (t + K < n_stages)"
                )
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

        for t in &inactive_stages {
            assert!(
                decision_at(*t).is_none(),
                "anticipated_decision_mw must be None at inactive stage t={t} \
             (t + K >= n_stages, K=3; strict-boundary predicate excludes this stage)",
            );
        }

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
}
mod anticipated_forward_pass {
    //! End-to-end integration test verifying the anticipated-state ring-buffer
    //! evolution across the full forward pass for a 5-stage K=2 system.
    //!
    //! Ring-buffer shift semantics: the LP's own in-LP definition rows resolve
    //! the ring transition — an interior slot's outgoing column is pinned to the
    //! next slot's incoming value (`slot[s]_out = slot[s+1]_in`), and the newest
    //! slot's outgoing column is pinned to the fresh decision
    //! (`slot[K-1]_out = decision_col`) — so slot 0 at stage `t+1` equals slot 1
    //! at stage `t`. The forward pass reads these outgoing columns by identity
    //! (`StateSpace::state_to_lp_column`), with no Rust-side shift step.

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
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroStageBounds, HydroStagePenalties,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
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

    /// Build a 5-stage system with one anticipated thermal (K=2, seeded
    /// `[100.0, 50.0]`) and one backup thermal that alone covers the 150 MW load, so
    /// the LP is always feasible.
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
                anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
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
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                max_diversion_m3s: None,
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
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
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
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: anticipated_id,
                values_mw: vec![100.0, 50.0],
            }],
            recent_observations: vec![],
            past_defluences: vec![],
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
            state_space: cobre_io::config::StateSpaceConfig::default(),
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
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

    /// Verify that the anticipated-state ring buffer evolves correctly across all
    /// 5 stages of the forward pass.
    ///
    /// The block layout is slot-major, plant-minor; with `n_anticipated=1`,
    /// `k_max=2` it is `[slot0_plant0, slot1_plant0]`. `state_at_capture` is filled
    /// by two paths, which is why slots 0 and 1 carry the same values:
    ///
    /// - **Stage 0**: forward pass writes the post-shift outgoing state of stage 0.
    /// - **Stages 1..=4**: backward pass stores the forward outgoing of stage `t-1`
    ///   as its trial point `x_hat` — so `basis_cache[1]` (trial point of stage 1)
    ///   equals `basis_cache[0]` (forward outgoing of stage 0).
    #[test]
    fn five_stage_k2_anticipated_state_ring_buffer_evolution() {
        let system = build_system();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        // One iteration populates basis_cache for every stage.
        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");

        assert!(
            outcome.error.is_none(),
            "training error must be None; got {:?}",
            outcome.error
        );
        assert!(
            outcome.result.iterations >= 1,
            "at least 1 iteration must complete; got {}",
            outcome.result.iterations
        );

        let state = setup.stage_state();
        let n_ant = state.n_anticipated;
        let k_max = state.k_max;
        assert_eq!(n_ant, 1, "fixture must have exactly 1 anticipated thermal");
        assert_eq!(k_max, 2, "fixture must have k_max = 2");

        let ant_start = state.anticipated_slots_out.start;

        let basis_cache = &outcome.result.basis_cache;
        assert_eq!(
            basis_cache.len(),
            5,
            "basis_cache must have one entry per study stage"
        );

        // Forward pass stores the post-shift outgoing state. Seed [100.0, 50.0]
        // shifts to slot 0 = 50.0 (no other code path produces this exact value);
        // slot 1 = LP decision d_0 ∈ [0, 100].
        let s0 = basis_cache[0]
            .as_ref()
            .expect("stage 0 basis must be captured")
            .state_at_capture
            .as_slice();
        assert!(
            (s0[ant_start] - 50.0).abs() < 1e-9,
            "stage 0 slot 0 must equal seeded slot 1 = 50.0; got {}",
            s0[ant_start]
        );
        assert!(
            (-0.01..=100.01).contains(&s0[ant_start + 1]),
            "stage 0 slot 1 (= d_0) must lie in [0, 100]; got {}",
            s0[ant_start + 1]
        );

        // Backward pass for stage 1 stores the forward outgoing of stage 0 as its
        // trial point x_hat, so basis_cache[1] equals basis_cache[0].
        let s1 = basis_cache[1]
            .as_ref()
            .expect("stage 1 basis must be captured")
            .state_at_capture
            .as_slice();
        assert!(
            (s1[ant_start] - s0[ant_start]).abs() < 1e-9,
            "stage 1 slot 0 ({}) must equal stage 0 slot 0 ({}) — both carry \
         the post-shift outgoing state of forward stage 0",
            s1[ant_start],
            s0[ant_start],
        );
        assert!(
            (s1[ant_start + 1] - s0[ant_start + 1]).abs() < 1e-9,
            "stage 1 slot 1 ({}) must equal stage 0 slot 1 ({}) — both carry \
         d_0 from the forward pass",
            s1[ant_start + 1],
            s0[ant_start + 1],
        );

        // For t≥2, basis_cache[t] holds the forward outgoing of stage t-1, so the
        // forward shift gives s_curr slot 0 == s_prev slot 1.
        for t in 2..5_usize {
            let s_curr = basis_cache[t]
                .as_ref()
                .unwrap_or_else(|| panic!("stage {t} basis must be captured"))
                .state_at_capture
                .as_slice();
            let s_prev = basis_cache[t - 1]
                .as_ref()
                .unwrap_or_else(|| panic!("stage {} basis must be captured", t - 1))
                .state_at_capture
                .as_slice();

            assert!(
                (s_curr[ant_start] - s_prev[ant_start + 1]).abs() < 1e-9,
                "stage {t} slot 0 ({}) must equal stage {} slot 1 ({})",
                s_curr[ant_start],
                t - 1,
                s_prev[ant_start + 1],
            );
        }

        for t in 0..5_usize {
            let s_t = basis_cache[t]
                .as_ref()
                .unwrap_or_else(|| panic!("stage {t} basis must be captured"))
                .state_at_capture
                .as_slice();
            let decision = s_t[ant_start + 1];
            assert!(
                (-0.01..=100.01).contains(&decision),
                "stage {t} slot 1 must lie in [0, 100]; got {decision}",
            );
        }
    }
}
mod anticipated_closed_form_lb_k1_single_thermal {
    //! Closed-form lower-bound canary for anticipated thermals.
    //!
    //! Two-stage, K=1, no hydro, no stochastic noise, single deterministic opening
    //! per stage. The lower bound returned by `train` is hand-derivable from the LP
    //! coefficients written by the `lp::builder` column/row fill paths, so any
    //! deviation from `EXPECTED_LB` flags a value-correctness regression that the
    //! larger structural-only integration tests cannot pin (no closed form exists
    //! for those fixtures).
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
    //!   The past must be zero so that any non-zero anticipated
    //!   delivery observed at stage 1 is attributable to the stage-0 decision.
    //! - 1 deterministic opening per stage.
    //! - Default `PolicyGraph::annual_discount_rate = 0.0`, so every
    //!   `discount_factors[t] = 1.0` and `cumulative_discount_factors[t] = 1.0`.
    //!
    //! ## Closed-form derivation
    //!
    //! The fishing constraint is always active for every anticipated plant at
    //! every stage, including stage 0. The fishing row `g_a_t − x_state_t = 0` is
    //! therefore emitted at stage 0 as well, and `fill_thermal_columns` skips the
    //! per-block cost of the anticipated column at stage 0 (never written, leaving
    //! it at zero; the anticipated thermal is detected via
    //! `anticipated_local_by_sys_pos`, same always-active path). See the K=1
    //! sign-chain table for the cut-coefficient sign convention that applies
    //! here.
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
    //! every stage including 0; per-block anticipated cost skipped in
    //! `fill_thermal_columns`, so it stays at zero):
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
    //! `fill_anticipated_columns` to
    //! `c_a · total_hours_per_stage[delivery=1] · cumulative_discount_factors[1] =
    //! c_a · 1 · 1 = c_a`.
    //!
    //! Stage 1 (delivery; fishing always active; decision
    //! `1 + 1 < 2` — FALSE, so `d_ant_1 ∈ [0,0]` and per-block anticipated cost
    //! is skipped in `fill_thermal_columns`, so it stays at zero):
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

    use cobre_core::entities::{bus::DeficitSegment, thermal::AnticipatedConfig};
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroStageBounds, HydroStagePenalties,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
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
    // Closed-form fixture parameters (single source of truth).
    // ---------------------------------------------------------------------------

    const N_STAGES: usize = 2;
    const K_MAX: usize = 1;
    const BLOCK_HOURS: f64 = 1.0;

    /// MW.
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

    /// $/MWh.
    const C_B: f64 = 100.0;

    /// Deficit cost ($/MWh). Set well above `C_B` so deficit is never preferred
    /// over backup dispatch.
    const C_DEFICIT: f64 = 1000.0;

    /// Closed-form lower bound under always-active fishing:
    /// `T* = (C_A + C_B) · D_LOAD = (10 + 100) · 50 = 5500.0` USD. See module docs
    /// for the full derivation.
    const EXPECTED_LB: f64 = (C_A + C_B) * D_LOAD; // = 5500.0

    const ANTICIPATED_ID: EntityId = EntityId(2);
    const BACKUP_ID: EntityId = EntityId(3);
    const BUS_ID: EntityId = EntityId(1);

    // SystemBuilder::build() sorts thermals by EntityId ascending, so the
    // global thermal indices end up: 0 → id=2 (anticipated), 1 → id=3 (backup).
    const THERMAL_IDX_ANT: usize = 0;
    const THERMAL_IDX_BACKUP: usize = 1;

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    fn build_system() -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = make_bus(
            BUS_ID,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: C_DEFICIT,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let thermal_ant = make_thermal(
            ANTICIPATED_ID,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: M_ANT,
                cost_per_mwh: C_A,
                anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            BACKUP_ID,
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: B_BACK,
                cost_per_mwh: C_B,
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
                mean_mw: D_LOAD,
                std_mw: 0.0,
            })
            .collect();

        // Overrides span the full thermal axis (n_stages + k_max) so
        // `fill_anticipated_columns` reads a well-defined cost at delivery (stage 1).
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
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 0.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                    max_diversion_m3s: None,
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
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
        );

        let thermal_axis = N_STAGES + K_MAX;
        for s in 0..thermal_axis {
            *bounds.thermal_bounds_mut(THERMAL_IDX_ANT, s) =
                ThermalStageBounds { cost_per_mwh: C_A };
            *bounds.thermal_block_base_mut(THERMAL_IDX_ANT, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: M_ANT,
            };
            *bounds.thermal_bounds_mut(THERMAL_IDX_BACKUP, s) =
                ThermalStageBounds { cost_per_mwh: C_B };
            *bounds.thermal_block_base_mut(THERMAL_IDX_BACKUP, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: B_BACK,
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

        // Past commitment is zero (the strict-zero precondition for
        // the K=1 ring buffer). Any non-zero delivery observed at stage 1 must
        // come from the stage-0 decision.
        let initial_conditions = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: ANTICIPATED_ID,
                values_mw: vec![0.0],
            }],
            recent_observations: vec![],
            past_defluences: vec![],
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

    /// Iteration limit = 4: three iterations reach the lower envelope at the kink
    /// `d_ant_0 = D`, plus one to settle degenerate-vertex selection. Pinned by the
    /// `iterations == 4` assertion.
    fn build_config() -> Config {
        Config {
            schema: None,
            state_space: cobre_io::config::StateSpaceConfig::default(),
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 4 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    // ---------------------------------------------------------------------------
    // Test
    // ---------------------------------------------------------------------------

    /// Closed-form canary: the LP-derived lower bound for the 2-stage K=1 fixture
    /// must equal `EXPECTED_LB = (C_A + C_B) · D_LOAD = 5500.0` bit-for-bit.
    #[test]
    fn anticipated_closed_form_lb_k1_single_thermal() {
        let system = build_system();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
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
        assert_eq!(result.iterations, 4, "iterations mismatch");

        let actual = result.final_lb;
        assert!(actual.is_finite(), "final_lb must be finite; got {actual}");
        assert_eq!(
            actual.to_bits(),
            EXPECTED_LB.to_bits(),
            "closed-form LB mismatch: actual = {actual}, expected = {EXPECTED_LB}. \
         Under always-active fishing the answer is `(C_A + C_B) · D_LOAD = 5500.0` \
         (stage-0 backup covers load; anticipated column cost skipped in \
         fill_thermal_columns, so it stays at zero). \
         If a libhighs upgrade introduces sub-ULP arithmetic drift, switch to \
         a 1e-12 relative tolerance check.",
        );

        let gap = result.final_gap;
        assert!(
            gap.is_finite() && gap.abs() < 1e-9,
            "final_gap should be approximately zero for a fully deterministic \
         fixture; got {gap}",
        );
    }
}
mod lead_time_single_decider_end_to_end {
    //! The first true `LeadTime` parse→validate→setup→train load-path exercise:
    //! a single-decider `LeadTime` thermal (`|C(t)| <= 1` everywhere) must solve
    //! end-to-end with no panic, validating both `template.rs`'s `StateSpace`
    //! threading and the in-LP ring for `LeadTime`.
    //!
    //! ## Fixture
    //!
    //! Same topology as `anticipated_closed_form_lb_k1_single_thermal`
    //! (1 anticipated thermal, 1 backup thermal, no hydro, always-active
    //! fishing), extended to 3 uniform 744h stages with
    //! `AnticipatedConfig::LeadTime(744.0)` in place of `LeadStages(1)`. On a
    //! uniform 744h calendar a 744h lead resolves to exactly the same
    //! decider as a constant 1-stage lead (`c(m) = m - 1`), so the closed-form
    //! derivation is the direct 3-stage extension of the 2-stage K=1 case:
    //!
    //! - Stage 0's delivery is the pre-study seed (`past_anticipated_commitments
    //!   = [0.0]`), so `g_a_0 = 0` and the backup covers `D` alone.
    //! - Stage 0 decides `d_ant_0`, delivered at stage 1; stage 1 decides
    //!   `d_ant_1`, delivered at stage 2 (the last stage decides nothing
    //!   further — its own `C(2)` is empty).
    //! - Each decision's sub-problem is independent
    //!   (`min_x c_a·x + c_b·max(0, D-x)` over `x ∈ [0, M]`) and — since
    //!   `c_a < c_b` — is minimised at `x = D` for both.
    //!
    //! `T* = H·[c_b·D + c_a·D + c_b·max(0,D-D) + c_a·D + c_b·max(0,D-D)]
    //!     = H·D·(c_b + 2·c_a)`.

    use cobre_core::entities::{bus::DeficitSegment, thermal::AnticipatedConfig};
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroStageBounds, HydroStagePenalties,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
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

    const N_STAGES: usize = 3;
    const K_MAX: usize = 1;
    const LEAD_TIME_HOURS: f64 = 744.0;
    const BLOCK_HOURS: f64 = 744.0;

    const D_LOAD: f64 = 50.0;
    const M_ANT: f64 = 100.0;
    const B_BACK: f64 = 200.0;
    const C_A: f64 = 10.0;
    const C_B: f64 = 100.0;
    const C_DEFICIT: f64 = 1000.0;

    /// Closed-form lower bound: `T* = H · D · (C_B + 2 · C_A)` — see module docs.
    const EXPECTED_LB: f64 = BLOCK_HOURS * D_LOAD * (C_B + 2.0 * C_A); // = 4_464_000.0

    const ANTICIPATED_ID: EntityId = EntityId(2);
    const BACKUP_ID: EntityId = EntityId(3);
    const BUS_ID: EntityId = EntityId(1);

    // SystemBuilder::build() sorts thermals by EntityId ascending, so the
    // global thermal indices end up: 0 → id=2 (anticipated), 1 → id=3 (backup).
    const THERMAL_IDX_ANT: usize = 0;
    const THERMAL_IDX_BACKUP: usize = 1;

    fn build_system() -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = make_bus(
            BUS_ID,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: C_DEFICIT,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let thermal_ant = make_thermal(
            ANTICIPATED_ID,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: M_ANT,
                cost_per_mwh: C_A,
                anticipated_config: Some(AnticipatedConfig::LeadTime(LEAD_TIME_HOURS)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            BACKUP_ID,
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: B_BACK,
                cost_per_mwh: C_B,
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
                mean_mw: D_LOAD,
                std_mw: 0.0,
            })
            .collect();

        // Overrides span the full thermal axis (n_stages + k_max) so
        // `fill_anticipated_columns` reads a well-defined cost at every delivery
        // stage.
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
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 0.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                    max_diversion_m3s: None,
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
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
        );

        let thermal_axis = N_STAGES + K_MAX;
        for s in 0..thermal_axis {
            *bounds.thermal_bounds_mut(THERMAL_IDX_ANT, s) =
                ThermalStageBounds { cost_per_mwh: C_A };
            *bounds.thermal_block_base_mut(THERMAL_IDX_ANT, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: M_ANT,
            };
            *bounds.thermal_bounds_mut(THERMAL_IDX_BACKUP, s) =
                ThermalStageBounds { cost_per_mwh: C_B };
            *bounds.thermal_block_base_mut(THERMAL_IDX_BACKUP, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: B_BACK,
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

        // Past commitment is zero (the strict-zero precondition mirroring the
        // K=1 ring-buffer fixture). Any non-zero delivery observed at stage 1
        // must come from the stage-0 decision.
        let initial_conditions = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: ANTICIPATED_ID,
                values_mw: vec![0.0],
            }],
            recent_observations: vec![],
            past_defluences: vec![],
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

    fn build_config() -> Config {
        Config {
            schema: None,
            state_space: cobre_io::config::StateSpaceConfig::default(),
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                // 2 iterations reach the closed-form LB; 2 more settle the
                // upper-bound gap to ~0 (mirrors the K=1 2-stage fixture's margin).
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 4 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// The first true `LeadTime` parse→validate→setup→train load-path exercise:
    /// a single-decider `LeadTime(744.0)` thermal on a uniform 3×744h calendar
    /// trains to convergence with no panic, matching the closed-form optimum
    /// `EXPECTED_LB = H · D · (C_B + 2 · C_A) = 4_464_000.0` USD.
    #[test]
    fn lead_time_single_decider_solves_end_to_end() {
        let system = build_system();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
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
        assert_eq!(result.iterations, 4, "iterations mismatch");

        let gap = result.final_gap;
        assert!(
            gap.is_finite() && gap.abs() < 1e-6,
            "final_gap should be approximately zero (LB == UB) for a fully \
             deterministic fixture; got {gap}",
        );

        let actual = result.final_lb;
        assert!(actual.is_finite(), "final_lb must be finite; got {actual}");
        let rel_diff = (actual - EXPECTED_LB).abs() / EXPECTED_LB.abs();
        assert!(
            rel_diff < 1e-9,
            "closed-form LB mismatch: actual = {actual}, expected = {EXPECTED_LB} \
             (rel_diff = {rel_diff}). See module docs for the derivation.",
        );
    }
}
mod anticipated_numerical_reconciliation_k2 {
    //! Numerical reconciliation test: LP total cost must match the analytical optimum
    //! for a K=2, 6-stage fixture with zero discount rate.
    //!
    //! A regression that silently zeroes intermediate-stage anticipated dispatch
    //! forces backup to carry that load, inflating the LP total by hundreds of times.
    //!
    //! ## Analytical optimum (discount rate = 0, all discount factors = 1.0)
    //!
    //! Parameters:
    //! - n_stages = 6, K = 2, load = 150 MW, block duration = 744 h/stage
    //! - Anticipated thermal: max_gen = 200 MW, cost = $10/MWh
    //! - Backup thermal: max_gen = 500 MW, cost = $5000/MWh
    //! - Excess generation cost = $0 (over-commitment is free)
    //!
    //! Stages partition into three zones:
    //!
    //! **Zone A — Active decision stages (t + K < n_stages → t ∈ {0, 1, 2, 3}):**
    //! The LP decides `d_t ∈ [0, 200]`. Because load = 150 MW < max_gen = 200 MW
    //! and excess is free, the LP commits `d_t = load = 150 MW` — not max_gen.
    //! Over-committing to 200 MW costs 50 MW × $10/MWh × 744 h = $372k/stage extra
    //! with no benefit. The per-block cost of the anticipated-decision column is
    //! $10/MWh, charged at the decision stage.
    //!
    //! Anticipated decision cost = 4 stages × 150 MW × 744 h × $10/MWh
    //!                           = **$4,464,000**
    //!
    //! **Zone B — Delivery stages with matured anticipated commitment (t ∈ {2, 3, 4, 5}):**
    //! The anticipated thermal delivers `committed_t = d_{t-K} = 150 MW = load`.
    //! Per-block cost on the anticipated thermal at delivery stages is skipped in
    //! `fill_thermal_columns` (never written; the anticipated thermal is detected
    //! via `anticipated_local_by_sys_pos`), so delivered generation costs $0
    //! in the objective. No backup needed since 150 MW = load exactly.
    //!
    //! **Zone C — Pre-horizon stages (t ∈ {0, 1}):**
    //! The always-active fishing predicate pins the anticipated thermal to seed
    //! slot 0 = 0 MW (`past_anticipated_commitments = [0.0, 0.0]`). The LP must
    //! dispatch backup at $5000/MWh to meet the 150 MW load. The cost-zeroing
    //! predicate is also always-active, so the anticipated thermal column has
    //! objective 0 — but its column upper bound is fishing-pinned to 0, leaving
    //! backup as the sole feasible source.
    //!
    //! Pre-horizon backup cost = 2 stages × 150 MW × 744 h × $5000/MWh
    //!                         = **$1,116,000,000**
    //!
    //! **Total analytical optimum = $4,464,000 + $1,116,000,000 = $1,120,464,000**
    //!
    //! The 5/6/7 entity IDs are distinct from the K=2 and K=3 saturation tests so
    //! combined nextest runs give unambiguous per-entity failure attribution.

    use cobre_io::config::{SimulationSelection, TrainingSelection};
    use std::sync::mpsc;

    use cobre_core::entities::{
        bus::DeficitSegment,
        hydro::{HydroGenerationModel, HydroPenalties},
        thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, PolicyGraph, PolicyGraphType, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroStageBounds, HydroStagePenalties,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
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
    // Analytical optimum constants (documented in module-level doc comment above)
    // ---------------------------------------------------------------------------

    /// Active decision cost: 4 stages × 150 MW (load, not max_gen — over-committing
    /// is wasted when excess is free) × 744 h × $10/MWh = $4,464,000.
    const EXPECTED_DECISION_COST_USD: f64 = 4.0 * 150.0 * 744.0 * 10.0;

    /// Pre-horizon backup cost: 2 stages × 150 MW × 744 h × $5000/MWh = $1,116,000,000.
    /// At t∈{0,1} the always-active fishing equality pins the anticipated thermal to
    /// the zero seed (slot 0 = 0 MW), leaving backup as the sole feasible source for
    /// the 150 MW load. See module doc Zone C.
    const EXPECTED_PRE_HORIZON_BACKUP_COST_USD: f64 = 2.0 * 150.0 * 744.0 * 5000.0;

    /// Total = active decision cost (stages 2..=5) + pre-horizon backup cost
    /// (stages 0, 1) = $4,464,000 + $1,116,000,000 = $1,120,464,000.
    const EXPECTED_TOTAL_USD: f64 =
        EXPECTED_DECISION_COST_USD + EXPECTED_PRE_HORIZON_BACKUP_COST_USD;

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// Build the 6-stage K=2 reconciliation fixture.
    ///
    /// The trivial hydro keeps the model in the thermal regime; it exists only so
    /// `n_hydros = 1` is satisfied without adding a hydro state variable that
    /// complicates interpretation. `annual_discount_rate = 0.0` collapses all
    /// discount factors to 1.0, making the analytical cost summation exact.
    fn build_system_reconciliation_k2() -> cobre_core::System {
        use chrono::NaiveDate;

        let k: usize = 2;
        let n_stages: usize = 6;

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

        let anticipated_id = EntityId(5);
        let thermal_ant = make_thermal(
            anticipated_id,
            ThermalSpec {
                name: "T_ant_reconcil".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
                cost_per_mwh: 10.0,
                anticipated_config: Some(AnticipatedConfig::LeadStages(k as u32)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            EntityId(6),
            ThermalSpec {
                name: "T_backup_reconcil".to_string(),
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

        let hydro = make_hydro(
            EntityId(7),
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
                hydro_id: EntityId(7),
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
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 1.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 1.0,
                max_diversion_m3s: None,
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

        // The padding region [n_stages, n_stages + k) is the delivery-stage axis
        // read by `fill_anticipated_columns`; it must carry the
        // per-thermal cost so the decision column's objective coefficient is
        // non-zero.
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
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 200.0,
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
        );
        // SystemBuilder sorts by EntityId: index 0 = anticipated (id=5), index 1 =
        // backup (id=6).
        for s in 0..thermal_axis {
            bounds.thermal_bounds_mut(0, s).cost_per_mwh = 10.0;
            bounds.thermal_block_base_mut(0, s).max_generation_mw = 200.0;
            bounds.thermal_bounds_mut(1, s).cost_per_mwh = 5000.0;
            bounds.thermal_block_base_mut(1, s).max_generation_mw = 500.0;
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

        // Zero seeds: slot 0 carries 0 MW at both pre-horizon stages (Zone C).
        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(7),
                value_hm3: 0.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: anticipated_id,
                values_mw: vec![0.0, 0.0],
            }],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        // Set explicitly (not relying on `PolicyGraph::default()`) so a future
        // default change cannot silently introduce NPV scaling into the analytical
        // derivation.
        let policy_graph = PolicyGraph {
            stage_discount_rate_overrides: std::collections::HashMap::new(),
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: vec![],
            nodes: Vec::new(),
            season_map: None,
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
            .policy_graph(policy_graph)
            .build()
            .expect("build_system_reconciliation_k2: valid system")
    }

    // ---------------------------------------------------------------------------
    // Config builder
    // ---------------------------------------------------------------------------

    /// Ten training iterations let the 500x cost asymmetry produce cuts that signal
    /// the value of anticipated dispatch, driving the observed cost to the optimum.
    fn build_config() -> Config {
        Config {
            schema: None,
            state_space: cobre_io::config::StateSpaceConfig::default(),
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 10 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig {
                enabled: true,
                io_channel_capacity: 8,
                selection: Some(SimulationSelection::Sampled { num_scenarios: 1 }),
                ..IoSimulationConfig::default()
            },
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    // ---------------------------------------------------------------------------
    // Test
    // ---------------------------------------------------------------------------

    /// Assert that the LP total cost equals the hand-derived analytical optimum
    /// ($1,120,464,000; derivation in module doc) for a K=2, 6-stage fixture with
    /// zero discount rate.
    #[test]
    fn lp_total_cost_matches_analytical_optimum_k2_discount_zero() {
        let system = build_system_reconciliation_k2();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new: must succeed");

        let outcome = setup
            .train(&mut solver, &comm, 10, ActiveSolver::new, None, None)
            .expect("training error: train() must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: training returned an error: {:?}",
            outcome.error,
        );

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("workspace pool error: create_workspace_pool must succeed");
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
            .expect("simulation error: simulate() must not return Err");
        drop(result_tx);
        let scenario_results = drain_handle.join().expect("drain thread must not panic");

        assert_eq!(
            scenario_results.len(),
            1,
            "simulation must stream exactly one scenario result",
        );
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            6,
            "scenario must contain one stage record per study stage (n_stages=6)",
        );

        // Sum `immediate_cost` (LP objective minus theta), excluding the future-cost
        // approximation — the realized cost the analytical optimum is derived for.
        let observed_total: f64 = scenario
            .stages
            .iter()
            .flat_map(|stage| stage.costs.iter())
            .map(|cost| cost.immediate_cost)
            .sum();

        // $1000 sits comfortably above HiGHS's 1e-9 precision yet far below the
        // ~$370M pre-fix error, giving 5+ orders of magnitude of detection headroom.
        const COST_TOLERANCE_USD: f64 = 1_000.0;
        assert!(
            (observed_total - EXPECTED_TOTAL_USD).abs() < COST_TOLERANCE_USD,
            "LP total cost {} differs from analytical optimum {} by {} \
         (tolerance ${:.2}). \
         Pre-fix behaviour: intermediate anticipated dispatch is zeroed \
         (d_1 = d_2 = d_3 ≈ 0), forcing the LP to backfill with backup at \
         $5000/MWh — producing a cost gap of approximately $370M.",
            observed_total,
            EXPECTED_TOTAL_USD,
            (observed_total - EXPECTED_TOTAL_USD).abs(),
            COST_TOLERANCE_USD,
        );

        // The named cost categories must sum to `immediate_cost` at every stage,
        // including the anticipated commitment fuel as `anticipated_thermal_cost`.
        // `hydro_violation_cost` already aggregates its six sub-components and
        // `spillage_cost` already includes diversion — sum the aggregates, not the
        // parts, or the total double-counts.
        const RECONCILE_TOLERANCE_USD: f64 = 1.0;
        let mut saw_nonzero_anticipated = false;
        for stage in &scenario.stages {
            for cost in &stage.costs {
                let category_sum = cost.thermal_cost
                    + cost.anticipated_thermal_cost
                    + cost.contract_cost
                    + cost.deficit_cost
                    + cost.excess_cost
                    + cost.storage_violation_cost
                    + cost.filling_target_cost
                    + cost.hydro_violation_cost
                    + cost.inflow_penalty_cost
                    + cost.generic_violation_cost
                    + cost.spillage_cost
                    + cost.turbined_cost
                    + cost.curtailment_cost
                    + cost.exchange_cost
                    + cost.pumping_cost;
                assert!(
                    (category_sum - cost.immediate_cost).abs() < RECONCILE_TOLERANCE_USD,
                    "stage {}: Σ(named cost categories) = {} must equal immediate_cost = {} \
                 (diff {}); the anticipated commitment fuel must be attributed to \
                 anticipated_thermal_cost, not left as an unattributed remainder",
                    cost.stage_id,
                    category_sum,
                    cost.immediate_cost,
                    (category_sum - cost.immediate_cost).abs(),
                );
                if cost.anticipated_thermal_cost.abs() > RECONCILE_TOLERANCE_USD {
                    saw_nonzero_anticipated = true;
                }
            }
        }
        // Zone A (decision stages t∈{0,1,2,3}) must book a positive
        // anticipated_thermal_cost; otherwise the new field is dead and the
        // reconciliation above would pass trivially.
        assert!(
            saw_nonzero_anticipated,
            "expected a non-zero anticipated_thermal_cost at the decision stages; \
         got zero everywhere (the GNL fuel was not attributed)",
        );
    }
}
mod anticipated_bridge_st_cruz_nova_k1 {
    //! Integration test: ST.CRUZ NOVA bridge-parity fixture with K=1 pre-horizon
    //! seed delivery.
    //!
    //! ## Analytical cost bound
    //!
    //! With `K = 1`, `n_stages = 5`, a single anticipated thermal (id=61,
    //! `ST_CRUZ_NOVA`), and `past_anticipated_commitments.values_mw = [204.5647]`:
    //!
    //! - Stage 0: anticipated delivers seed (204.5647 MW, zero cost) + backup
    //!   covers remaining (250.0 - 204.5647) MW at $5000/MWh × 744 h ≈ $169,019,316.
    //! - Stages 1–4 delivery: anticipated covers ≥ load (zeroed cost).
    //! - Decision cost ≤ 4 × 350 MW × $10/MWh × 744 h = $10,416,000.
    //! - Total ≤ $169,019,316 + $10,416,000 + $1,000 (tolerance).
    //!
    //! The 204.5647 MW seed is the block-fraction-weighted aggregate of ST.CRUZ NOVA
    //! per-block MW values (227.86, 238.37, 173.51) against September 2024 block
    //! fractions (0.2333, 0.2833, 0.4834); the 1e-3 MW tolerance reflects its
    //! four-decimal accuracy.
    //!
    //! The fishing constraint is always active for every anticipated plant, so a
    //! fishing row is emitted at every stage. The anticipated plant's delivery-stage
    //! per-block thermal cost is skipped in `fill_thermal_columns` (the plant is
    //! detected via `anticipated_local_by_sys_pos`), so those columns are consumed
    //! at zero cost.
    //!
    //! The 60-series entity IDs are distinct from the other anticipated tests so
    //! combined nextest runs give unambiguous per-entity failure attribution.

    use cobre_io::config::{SimulationSelection, TrainingSelection};
    use std::sync::mpsc;

    use cobre_core::entities::{
        bus::DeficitSegment,
        hydro::{HydroGenerationModel, HydroPenalties},
        thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, PolicyGraph, PolicyGraphType, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroStageBounds, HydroStagePenalties,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
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
    // Analytical cost bound constants (documented in module doc comment above)
    // ---------------------------------------------------------------------------

    const STAGE_0_BACKUP_COST_USD: f64 = (250.0 - 204.5647) * 744.0 * 5000.0;
    const MAX_DECISION_COST_USD: f64 = 4.0 * 350.0 * 744.0 * 10.0;
    const COST_TOLERANCE_USD: f64 = 1_000.0;
    const EXPECTED_TOTAL_UPPER_BOUND_USD: f64 =
        STAGE_0_BACKUP_COST_USD + MAX_DECISION_COST_USD + COST_TOLERANCE_USD;

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// Build the 5-stage K=1 ST.CRUZ NOVA fixture.
    ///
    /// Builds the resolved `cobre_core::System` directly via `SystemBuilder::new()`,
    /// bypassing the `cobre-io` parse-and-validate pipeline, to keep the test
    /// self-contained; the 204.5647 MW seed is within bounds and would also pass
    /// `load_case`.
    ///
    /// The anticipated ($10/MWh) vs backup ($5000/MWh) 500× ratio makes the LP
    /// prefer anticipated dispatch. `annual_discount_rate = 0.0` collapses all
    /// discount factors to 1.0, making the analytical cost derivation exact.
    fn build_system() -> cobre_core::System {
        use chrono::NaiveDate;

        let k: usize = 1;
        let n_stages: usize = 5;

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

        let anticipated_id = EntityId(61);
        let thermal_ant = make_thermal(
            anticipated_id,
            ThermalSpec {
                name: "ST_CRUZ_NOVA".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 350.0,
                cost_per_mwh: 10.0,
                anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            EntityId(62),
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

        // Trivial hydro keeps the model in the thermal regime; present only so
        // `n_hydros = 1` exercises the hydro state path without adding uncertainty.
        let hydro = make_hydro(
            EntityId(60),
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
                        start_date: NaiveDate::from_ymd_opt(2024, 9, 1).unwrap(),
                        end_date: NaiveDate::from_ymd_opt(2024, 10, 1).unwrap(),
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
                hydro_id: EntityId(60),
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
                mean_mw: 250.0,
                std_mw: 0.0,
            })
            .collect();

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 1.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 1.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 1.0,
                max_diversion_m3s: None,
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

        // The padding region [n_stages, n_stages + k) is the delivery-stage axis
        // read by `fill_anticipated_columns`; it must carry per-thermal
        // costs so the decision column's objective coefficient is non-zero.
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
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 350.0,
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
        );
        // SystemBuilder sorts by EntityId: index 0 = anticipated (id=61), index 1 =
        // backup (id=62). Without these per-thermal overrides the LP has no cost
        // incentive to commit anticipated capacity, so decision_at(t) collapses to
        // zero and masks the regression assertion.
        for s in 0..thermal_axis {
            bounds.thermal_bounds_mut(0, s).cost_per_mwh = 10.0;
            bounds.thermal_block_base_mut(0, s).max_generation_mw = 350.0;
            bounds.thermal_bounds_mut(1, s).cost_per_mwh = 5000.0;
            bounds.thermal_block_base_mut(1, s).max_generation_mw = 500.0;
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

        // Slot 0 = 204.5647 MW seed (block-fraction-weighted aggregate; see module
        // doc). The always-active fishing equality reads it at stage 0 and delivers
        // it at zero LP cost, leaving backup to cover the remainder.
        let initial_conditions = InitialConditions {
            storage: vec![HydroStorage {
                hydro_id: EntityId(60),
                value_hm3: 0.0,
            }],
            filling_storage: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: anticipated_id,
                values_mw: vec![204.5647],
            }],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        let policy_graph = PolicyGraph {
            stage_discount_rate_overrides: std::collections::HashMap::new(),
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: vec![],
            nodes: Vec::new(),
            season_map: None,
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
            .policy_graph(policy_graph)
            .build()
            .expect("build_system: valid system")
    }

    // ---------------------------------------------------------------------------
    // Config builder
    // ---------------------------------------------------------------------------

    /// One training iteration suffices: the stage-0 fishing equality pins the seed
    /// delivery regardless of cut quality, and the cost bound is deliberately
    /// generous to absorb the loose 1-iteration cut.
    fn build_config() -> Config {
        Config {
            schema: None,
            state_space: cobre_io::config::StateSpaceConfig::default(),
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig {
                enabled: true,
                io_channel_capacity: 8,
                selection: Some(SimulationSelection::Sampled { num_scenarios: 1 }),
                ..IoSimulationConfig::default()
            },
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    // ---------------------------------------------------------------------------
    // Test
    // ---------------------------------------------------------------------------

    /// Verify that the ST.CRUZ NOVA pre-horizon seed (204.5647 MW) is delivered at
    /// stage 0 via the always-active fishing predicate, and that the ring-buffer
    /// shift propagates in-study decisions correctly for stages 1–4.
    #[test]
    fn pre_horizon_seed_delivers_at_stage_zero_st_cruz_nova_k1() {
        let system = build_system();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train: must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error
        );

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("create_workspace_pool: must succeed");
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
            .expect("simulate: must not return Err");

        drop(result_tx);
        let scenario_results = drain_handle.join().expect("drain thread must not panic");

        assert_eq!(
            scenario_results.len(),
            1,
            "simulation must stream exactly one scenario result",
        );
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            5,
            "scenario must contain one record per study stage (n_stages=5)",
        );

        let anticipated_thermal_id: i32 = 61;
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

        let c0 = committed_at(0).expect(
            "AC-delivery FAIL: committed_at(0) is None. \
         The always-active fishing predicate is not delivering the 204.5647 MW seed \
         at stage 0. Legacy behaviour: predicate `K_i > stage_idx` gated \
         fishing off at stage 0 with K=1.",
        );
        assert!(
            (c0 - 204.5647).abs() < 1e-3,
            "AC-delivery FAIL: committed_at(0) = {c0} MW, expected 204.5647 MW within \
         1e-3 MW tolerance. The fishing equality at stage 0 must pin the anticipated \
         thermal to slot 0 = 204.5647 MW of the ring buffer.",
        );

        for t in 0..4_usize {
            let dt = decision_at(t).unwrap_or_else(|| {
                panic!(
                    "AC-decision-nonzero FAIL: decision_at({t}) is None; anticipated \
                 thermal id=61 was not found in stage {t} thermals or \
                 anticipated_decision_mw is absent (stage {t} is an active-decision \
                 stage: {t} + 1 < 5)",
                )
            });
            assert!(
                dt.abs() > 1e-6,
                "AC-decision-nonzero FAIL: decision_at({t}) = {dt} MW is zero (≤ 1e-6). \
             The LP should commit a non-trivial anticipated amount at stage {t} \
             to avoid $5000/MWh backup at the delivery stage.",
            );
            assert!(
                dt <= 350.0 + 1e-6,
                "AC-decision-nonzero FAIL: decision_at({t}) = {dt} MW exceeds \
             max_gen=350 MW. This indicates a bounds violation in the LP.",
            );
        }

        // After the shift ending stage t-1, slot 0 holds that stage's decision, and
        // stage t's fishing equality pins generation to it.
        for t in 1..5_usize {
            let ct = committed_at(t).unwrap_or_else(|| {
                panic!(
                    "AC-ring-buffer FAIL: committed_at({t}) is None; expected a matured \
                 commitment from decision at stage {}",
                    t - 1,
                )
            });
            let d_prev = decision_at(t - 1).unwrap_or_else(|| {
                panic!(
                    "AC-ring-buffer FAIL: decision_at({}) is None (needed to check \
                 ring-buffer invariant at stage {t})",
                    t - 1,
                )
            });
            assert!(
                (ct - d_prev).abs() < 1e-6,
                "AC-ring-buffer FAIL: committed_at({t}) = {ct} MW should equal \
             decision_at({}) = {d_prev} MW (within 1e-6 MW). The ring buffer is \
             not correctly propagating in-study decisions.",
                t - 1,
            );
        }

        // Sum immediate_cost (LP objective minus theta), not total_cost — total_cost
        // includes the theta approximation artefact.
        let observed_total: f64 = scenario
            .stages
            .iter()
            .flat_map(|st| st.costs.iter().map(|c| c.immediate_cost))
            .sum();

        assert!(
            observed_total <= EXPECTED_TOTAL_UPPER_BOUND_USD,
            "AC-cost-bound FAIL: observed_total = ${observed_total:.2} exceeds upper \
         bound ${EXPECTED_TOTAL_UPPER_BOUND_USD:.2}. \
         Breakdown: STAGE_0_BACKUP_COST_USD=${STAGE_0_BACKUP_COST_USD:.2}, \
         MAX_DECISION_COST_USD=${MAX_DECISION_COST_USD:.2}, \
         COST_TOLERANCE_USD=${COST_TOLERANCE_USD:.2}. \
         If the seed is not delivered (legacy predicate), stage-0 backup covers \
         250 MW instead of 45.4353 MW, producing ~$927M >> this bound.",
        );
    }
}
mod anticipated_convergence_slow {
    //! Slow-gated convergence test for a medium anticipated-thermal case (fixture
    //! built by `build_system`).
    //!
    //! Verifies that `SamplingScheme::OutOfSample` converges to a lower bound
    //! within 10% relative tolerance of the `SamplingScheme::InSample` lower
    //! bound when anticipated thermals are present. The 10% tolerance (relaxed from
    //! 5%) accommodates the anticipated ring-buffer variance source that is absent
    //! in the pure-hydro convergence test.

    use chrono::NaiveDate;
    use cobre_core::entities::{
        bus::DeficitSegment,
        hydro::{HydroGenerationModel, HydroPenalties},
        thermal::AnticipatedConfig,
    };
    use cobre_core::scenario::{InflowModel, LoadModel, SamplingScheme, ScenarioSource};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroStageBounds, HydroStagePenalties,
        HydroStorage, InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_sddp::{
        InflowNonNegativityMethod, StoppingMode, StoppingRule, StoppingRuleSet, StudySetup,
        TrainingOutcome,
        hydro_models::PrepareHydroModelsResult,
        setup::{ConstructionConfig, SimulationEnumeratedRequest},
    };
    use cobre_solver::ActiveSolver;
    use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

    use super::common::StubComm;
    use super::common::builders::{
        BusSpec, HydroSpec, StageSpec, ThermalSpec, make_bus, make_hydro, make_stage, make_thermal,
    };

    // ---------------------------------------------------------------------------
    // System builder
    // ---------------------------------------------------------------------------

    /// Build the 4-stage K=2 convergence fixture (1 hydro, 1 anticipated thermal,
    /// 1 backup). The LP is always feasible: the backup thermal alone covers the
    /// 220 MW load.
    fn build_system(branching_factor: usize) -> cobre_core::System {
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
                anticipated_config: Some(AnticipatedConfig::LeadStages(2)),
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

        let n_stages = 4_usize;
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
                            branching_factor,
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
                mean_mw: 220.0,
                std_mw: 0.0,
            })
            .collect();

        let k_max: usize = 2;
        let n_st = n_stages;

        fn default_hydro_bounds() -> HydroStageBounds {
            HydroStageBounds {
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            }
        }

        fn default_hydro_block_bounds() -> HydroBlockBounds {
            HydroBlockBounds {
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                max_diversion_m3s: None,
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
                hydro_block: default_hydro_block_bounds(),
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
                thermal_block: ThermalBlockBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
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
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: anticipated_id,
                values_mw: vec![40.0, 20.0],
            }],
            recent_observations: vec![],
            past_defluences: vec![],
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
    // Training runner
    // ---------------------------------------------------------------------------

    /// Run the SDDP training pipeline programmatically.
    ///
    /// Builds a `StudySetup` using the low-level `from_broadcast_params` path
    /// (identical to `forward_sampler_integration.rs::run_programmatic`) so that
    /// both the `InSample` and `OutOfSample` sampling schemes can be exercised
    /// without going through the file-based config/IO path.
    fn run_training(
        system: &cobre_core::System,
        sampling_scheme: SamplingScheme,
        forward_seed: Option<i64>,
        inflow_method: InflowNonNegativityMethod,
    ) -> TrainingOutcome {
        const FORWARD_PASSES: u32 = 10;
        const MAX_ITERATIONS: u64 = 30;

        let tree_seed: u64 = 42;
        // `forward_seed` for `build_stochastic_context` is the unsigned abs of the
        // signed seed from `ScenarioSource`, matching the conversion in
        // `forward_sampler_integration.rs::run_programmatic`.
        let fwd_seed_u64 = forward_seed.map(i64::unsigned_abs);

        let stochastic = build_stochastic_context(
            system,
            tree_seed,
            fwd_seed_u64,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .expect("build_stochastic_context must succeed");

        let hydro_models = PrepareHydroModelsResult::default_from_system(system);

        let stopping_rule_set = StoppingRuleSet {
            rules: vec![StoppingRule::IterationLimit {
                limit: MAX_ITERATIONS,
            }],
            mode: StoppingMode::Any,
        };

        let config = ConstructionConfig {
            seed: tree_seed,
            forward_passes: FORWARD_PASSES,
            training_enumerated: false,
            stopping_rule_set,
            n_scenarios: 0,
            simulation_enumerated: SimulationEnumeratedRequest::Sampled,
            io_channel_capacity: 0,
            policy_path: String::new(),
            inflow_method,
            cut_selection: None,
            cut_activity_tolerance: 0.0,
            budget: None,
            export_states: false,
            scalar_parameters: Vec::new(),
            training_solver_backward: None,
            training_solver_forward: None,
            simulation_solver: None,
            backward_scheduler: cobre_io::config::BackwardScheduler::default(),
            cost_scale_factor: cobre_sddp::DEFAULT_COST_SCALE_FACTOR,
            inflow_lag_depth: None,
        };

        let source = ScenarioSource {
            inflow_scheme: sampling_scheme,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            seed: forward_seed,
            historical_years: None,
        };

        let mut setup = StudySetup::from_broadcast_params(
            system,
            stochastic,
            config,
            hydro_models,
            &source,
            &source,
        )
        .expect("StudySetup::from_broadcast_params must succeed");

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

        setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok")
    }

    // ---------------------------------------------------------------------------
    // Slow-gated convergence test
    // ---------------------------------------------------------------------------

    /// Verify that `SamplingScheme::OutOfSample` converges to a lower bound within
    /// 10% relative tolerance of the `SamplingScheme::InSample` lower bound on a
    /// 4-stage K=2 anticipated-thermal system with `branching_factor=5` SAA noise.
    ///
    /// The 10% tolerance (relaxed from the 5% used in the pure-hydro convergence
    /// test) accommodates the additional variance introduced by the anticipated
    /// ring-buffer's interaction with the out-of-sample noise trajectories.
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    #[test]
    fn anticipated_k2_insample_vs_out_of_sample_convergence() {
        const BRANCHING_FACTOR: usize = 5;
        const RELATIVE_TOLERANCE: f64 = 0.10;

        let system_insample = build_system(BRANCHING_FACTOR);
        let system_oos = build_system(BRANCHING_FACTOR);

        let result_insample = run_training(
            &system_insample,
            SamplingScheme::InSample,
            None,
            InflowNonNegativityMethod::None,
        );

        assert!(result_insample.error.is_none(), "InSample error");
        let lb_insample = result_insample.result.final_lb;
        assert!(lb_insample > 0.0 && lb_insample.is_finite());

        let result_oos = run_training(
            &system_oos,
            SamplingScheme::OutOfSample,
            Some(42),
            InflowNonNegativityMethod::Truncation,
        );

        assert!(result_oos.error.is_none(), "OutOfSample error");
        let lb_oos = result_oos.result.final_lb;
        assert!(lb_oos > 0.0 && lb_oos.is_finite());

        let relative_error = (lb_oos - lb_insample).abs() / lb_insample.abs().max(1e-10);
        assert!(
            relative_error < RELATIVE_TOLERANCE,
            "convergence exceeded tolerance: {:.2}% vs {:.0}%",
            relative_error * 100.0,
            RELATIVE_TOLERANCE * 100.0,
        );
    }
}
mod a1b_value_cut_identity_anchor {
    //! LeadTime == LeadStages value/cut-identity anchor on a uniform calendar.
    //!
    //! On a uniform 3x744h calendar a physical lead of exactly one stage's hours
    //! (`LeadTime(744.0)`) and a one-stage-count lead (`LeadStages(1)`) resolve to
    //! the identical delivery-anchored decider `[None, Some(0), Some(1)]`, so both
    //! build a byte-identical LP and MUST train to bit-identical value, states, and
    //! anticipated-ring cuts. This module pins that equivalence end-to-end
    //! (train + simulate) as the A-1(b) value/cut-identity anchor.

    use cobre_core::entities::{bus::DeficitSegment, thermal::AnticipatedConfig};
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, PolicyGraph, PolicyGraphType, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroStageBounds, HydroStagePenalties,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_io::config::{SimulationSelection, TrainingSelection};
    use cobre_sddp::{SimulationScenarioResult, StudySetup};
    use cobre_solver::ActiveSolver;

    use std::sync::mpsc;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal,
    };

    const N_STAGES: usize = 3;
    const K_MAX: usize = 1;
    const STAGE_HOURS: f64 = 744.0;
    const LOAD_MW: f64 = 200.0;

    const BACKUP_COST: f64 = 100.0;
    const BACKUP_CAP: f64 = 200.0;
    const ANT_COST: f64 = 5.0;

    // Stage-varying anticipated delivery caps, indexed by delivery stage over the
    // `N_STAGES + K_MAX` axis `fill_anticipated_columns` reads. Stage 0 is the
    // seed-delivery stage AND the decision-anchored decoy for the delivery-1
    // decision; the delivery-stage-1 (150) and delivery-stage-2 (80) caps differ,
    // which is what makes the delivery-anchoring mutation non-vacuous — constant caps
    // would make a decision-anchored read indistinguishable from a delivery-anchored
    // one.
    const THERMAL_AXIS: usize = N_STAGES + K_MAX;
    const DELIVERY_CAP: [f64; THERMAL_AXIS] = [150.0, 150.0, 80.0, 80.0];

    // EntityId ordering: bus 1, backup T0 = 2, anticipated T1 = 3. System::build
    // sorts thermals by EntityId, so thermal_idx 0 = backup, 1 = anticipated.
    const BUS_ID: EntityId = EntityId(1);
    const BACKUP_ID: EntityId = EntityId(2);
    const ANT_ID: EntityId = EntityId(3);
    const THERMAL_IDX_BACKUP: usize = 0;
    const THERMAL_IDX_ANT: usize = 1;

    const ITERATIONS: usize = 20;

    fn build_system(anticipated_config: AnticipatedConfig) -> cobre_core::System {
        use chrono::NaiveDate;
        let date = |m: u32, d: u32| NaiveDate::from_ymd_opt(2024, m, d).expect("valid date");

        let bus = make_bus(
            BUS_ID,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: date(1, 1),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 1000.0,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let backup = make_thermal(
            BACKUP_ID,
            ThermalSpec {
                name: "T0_backup".to_string(),
                operational_start_date: date(1, 2),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: BACKUP_CAP,
                cost_per_mwh: BACKUP_COST,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let anticipated = make_thermal(
            ANT_ID,
            ThermalSpec {
                name: "T1_anticipated".to_string(),
                operational_start_date: date(1, 3),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: BACKUP_CAP,
                cost_per_mwh: ANT_COST,
                anticipated_config: Some(anticipated_config),
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
                        start_date: date(1, 1),
                        end_date: date(2, 1),
                        season_id: None,
                        blocks: vec![Block {
                            index: 0,
                            name: "S".to_string(),
                            duration_hours: STAGE_HOURS,
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
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 0.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                    max_diversion_m3s: None,
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
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
        );

        // fill_anticipated_columns reads the delivery cell at `stage_idx + K`, so
        // per-thermal bounds must cover the full `[0, N_STAGES + K_MAX)` axis. The
        // backup is constant; the anticipated plant carries the stage-varying caps.
        for s in 0..THERMAL_AXIS {
            *bounds.thermal_bounds_mut(THERMAL_IDX_BACKUP, s) = ThermalStageBounds {
                cost_per_mwh: BACKUP_COST,
            };
            *bounds.thermal_block_base_mut(THERMAL_IDX_BACKUP, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: BACKUP_CAP,
            };
            *bounds.thermal_bounds_mut(THERMAL_IDX_ANT, s) = ThermalStageBounds {
                cost_per_mwh: ANT_COST,
            };
            *bounds.thermal_block_base_mut(THERMAL_IDX_ANT, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: DELIVERY_CAP[s],
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

        // K=1 ring: exactly one pre-study delivery stage (m=0, decider None) in both
        // modes. Seed it to 0 MW so stage 0 delivers nothing from T1 and the
        // hand-derived dispatch lives entirely at delivery stages 1 and 2.
        let initial_conditions = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: ANT_ID,
                values_mw: vec![0.0],
            }],
            recent_observations: vec![],
            past_defluences: vec![],
        };

        let policy_graph = PolicyGraph {
            stage_discount_rate_overrides: std::collections::HashMap::new(),
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.0,
            transitions: vec![],
            nodes: Vec::new(),
            season_map: None,
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![backup, anticipated])
            .stages(stages)
            .load_models(load_models)
            .bounds(bounds)
            .penalties(penalties)
            .initial_conditions(initial_conditions)
            .policy_graph(policy_graph)
            .build()
            .expect("build_system: valid")
    }

    fn build_config(iterations: usize) -> Config {
        Config {
            schema: None,
            state_space: cobre_io::config::StateSpaceConfig::default(),
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                    limit: iterations as u32,
                }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig {
                enabled: true,
                io_channel_capacity: 8,
                selection: Some(SimulationSelection::Sampled { num_scenarios: 1 }),
                ..IoSimulationConfig::default()
            },
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// Converged run outputs for the two-mode identity comparison.
    struct RunOutputs {
        final_lb: f64,
        final_ub: f64,
        scenarios: Vec<SimulationScenarioResult>,
    }

    /// Train `iterations` then simulate one scenario, capturing the converged
    /// lower/upper bound before the outcome's basis cache is consumed by simulate.
    fn train_and_simulate(setup: &mut StudySetup, iterations: usize) -> RunOutputs {
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");
        let outcome = setup
            .train(
                &mut solver,
                &comm,
                iterations,
                ActiveSolver::new,
                None,
                None,
            )
            .expect("train must not return Err");
        assert!(
            outcome.error.is_none(),
            "training error must be None; got {:?}",
            outcome.error
        );
        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("create_workspace_pool");
        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (tx, rx) = mpsc::sync_channel(io_capacity);
        let drain = std::thread::spawn(move || rx.into_iter().collect::<Vec<_>>());
        setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &tx,
                None,
                None,
                &outcome.result.basis_cache,
            )
            .expect("simulate must not return Err");
        drop(tx);
        let scenarios = drain.join().expect("drain thread must not panic");

        RunOutputs {
            final_lb: outcome.result.final_lb,
            final_ub: outcome.result.final_ub,
            scenarios,
        }
    }

    /// Per-stage active cuts as `(slot, intercept, coefficients)`. With no hydro the
    /// full coefficient vector IS the anticipated ring (`n_state ==
    /// anticipated_slots_out.end`), so equality here is anticipated-ring cut identity.
    fn collect_cuts(setup: &StudySetup) -> Vec<Vec<(usize, f64, Vec<f64>)>> {
        (0..setup.fcf.pools.len())
            .map(|stage| {
                setup
                    .fcf
                    .active_cuts(stage)
                    .map(|(slot, intercept, coeffs)| (slot, intercept, coeffs.to_vec()))
                    .collect()
            })
            .collect()
    }

    fn committed_mw(scenario: &SimulationScenarioResult, t: usize) -> f64 {
        scenario.stages[t]
            .thermals
            .iter()
            .find(|th| th.thermal_id == ANT_ID.0)
            .and_then(|th| th.anticipated_committed_mw)
            .unwrap_or_else(|| panic!("committed_mw missing at stage {t}"))
    }

    fn decision_mw(scenario: &SimulationScenarioResult, t: usize) -> Option<f64> {
        scenario.stages[t]
            .thermals
            .iter()
            .find(|th| th.thermal_id == ANT_ID.0)
            .and_then(|th| th.anticipated_decision_mw)
    }

    fn backup_mw(scenario: &SimulationScenarioResult, t: usize) -> f64 {
        scenario.stages[t]
            .thermals
            .iter()
            .find(|th| th.thermal_id == BACKUP_ID.0)
            .map_or_else(
                || panic!("backup generation missing at stage {t}"),
                |th| th.generation_mw,
            )
    }

    /// `LeadTime(744h) == LeadStages(1)` on a uniform 3x744h calendar: the two modes
    /// resolve to the identical decider `[None, Some(0), Some(1)]`, build a
    /// byte-identical LP, and MUST train to bit-identical value, per-stage
    /// anticipated committed/decision MW, and anticipated-ring cut coefficients.
    ///
    /// This asserts SOLUTIONS (final LB/UB, committed/decision MW, cut coefficients),
    /// never LP bytes or a basis-dependent dual: the two modes' LPs are identical, so
    /// the optimum is bit-identical and `==` is the correct comparison. Runs under
    /// whichever backend the binary is compiled with, so the identity is pinned on
    /// both HiGHS and CLP via the per-backend CI matrix.
    ///
    /// Hand-derived dispatch (T1 cheap at $5/MWh runs to its delivery-stage cap; T0
    /// backup at $100/MWh fills the 200 MW load): delivery stage 1 -> T1 commits 150
    /// (its stage-1 cap), T0 supplies 50; delivery stage 2 -> T1 commits 80 (its
    /// stage-2 cap), T0 supplies 120.
    ///
    /// Delivery-anchoring is MUTATION-verified. Changing the production read in
    /// `fill_anticipated_columns` from `thermal_block_base(thermal_idx, delivery_stage)`
    /// to `thermal_block_base(thermal_idx, stage_idx)` (the decision stage) relaxes the
    /// stage-1 decision column to stage-1's 150 MW cap for delivery at stage 2; stage
    /// 2's own generation cap is 80 MW and fishing pins gen == committed, so the
    /// delivered 150 MW is undeliverable — the forward solve turns infeasible and the
    /// run errors (the capacity-drop infeasibility the delivery-anchoring contract
    /// forbids), failing this test at the `train`/`simulate` expect. The stage-1 vs
    /// stage-2 cap difference (150 vs 80) is what makes the mutation observable;
    /// constant caps would make it vacuous.
    #[test]
    fn a1b_lead_time_equals_lead_stages_uniform_calendar() {
        let mut setup_stages = build_setup_in_code(
            build_system(AnticipatedConfig::LeadStages(1)),
            &build_config(ITERATIONS),
        );
        let mut setup_time = build_setup_in_code(
            build_system(AnticipatedConfig::LeadTime(STAGE_HOURS)),
            &build_config(ITERATIONS),
        );

        let (k_max_stages, n_ant_stages) = {
            let s = setup_stages.stage_state();
            (s.k_max, s.n_anticipated)
        };
        let (k_max_time, n_ant_time) = {
            let s = setup_time.stage_state();
            (s.k_max, s.n_anticipated)
        };
        assert_eq!(n_ant_stages, 1, "one anticipated plant (LeadStages)");
        assert_eq!(n_ant_time, 1, "one anticipated plant (LeadTime)");
        assert_eq!(
            k_max_stages, k_max_time,
            "both modes must derive the same ring depth on the uniform calendar",
        );
        assert_eq!(k_max_stages, K_MAX, "ring depth must be 1");

        let run_stages = train_and_simulate(&mut setup_stages, ITERATIONS);
        let run_time = train_and_simulate(&mut setup_time, ITERATIONS);

        // ── Two-mode bit-identity ────────────────────────────────────────────────
        assert_eq!(
            run_stages.final_lb, run_time.final_lb,
            "final lower bound must be bit-identical across LeadStages/LeadTime",
        );
        assert_eq!(
            run_stages.final_ub, run_time.final_ub,
            "objective (final upper bound) must be bit-identical across modes",
        );

        let scen_stages = &run_stages.scenarios[0];
        let scen_time = &run_time.scenarios[0];
        assert_eq!(scen_stages.stages.len(), N_STAGES);
        assert_eq!(scen_time.stages.len(), N_STAGES);
        for t in 0..N_STAGES {
            assert_eq!(
                committed_mw(scen_stages, t),
                committed_mw(scen_time, t),
                "committed_mw at stage {t} must be bit-identical across modes",
            );
            assert_eq!(
                decision_mw(scen_stages, t),
                decision_mw(scen_time, t),
                "decision_mw at stage {t} must be bit-identical across modes",
            );
        }

        assert_eq!(
            collect_cuts(&setup_stages),
            collect_cuts(&setup_time),
            "anticipated-ring cut coefficients must be bit-identical across modes",
        );

        // ── Hand-derived dispatch (asserted on the LeadStages run; the identity
        //    above carries every equality to the LeadTime run) ────────────────────
        const TOL: f64 = 1e-6;
        assert!(
            (committed_mw(scen_stages, 1) - 150.0).abs() < TOL,
            "delivery stage 1: T1 must commit its 150 MW stage-1 cap; got {}",
            committed_mw(scen_stages, 1),
        );
        assert!(
            (backup_mw(scen_stages, 1) - 50.0).abs() < TOL,
            "delivery stage 1: T0 must supply 50 MW; got {}",
            backup_mw(scen_stages, 1),
        );
        assert!(
            (committed_mw(scen_stages, 2) - 80.0).abs() < TOL,
            "delivery stage 2: T1 must commit its 80 MW stage-2 cap (delivery-anchored, \
             not the 150 MW decision-stage cap); got {}",
            committed_mw(scen_stages, 2),
        );
        assert!(
            (backup_mw(scen_stages, 2) - 120.0).abs() < TOL,
            "delivery stage 2: T0 must supply 120 MW; got {}",
            backup_mw(scen_stages, 2),
        );
    }
}
mod a1c_stage_count_mode_anchor {
    //! Stage-count-mode (`LeadStages`) backwards-compatibility anchor on the d37
    //! unequal-hours monthly calendar `[730, 730, 730, 720, 744, 720]`.
    //!
    //! `LeadStages(ℓ)` is a pure index shift `c(m) = m − ℓ` whose result never
    //! reads the hour clock — the guarantee that shipped stage-count configs keep
    //! working unchanged on any calendar. The three resolver-level tests pin that
    //! the clock is never consulted in stage-count mode and contrast it with the
    //! physical `LeadTime` mode, which does consult it. The end-to-end test proves
    //! a single-decider `LeadTime` on this same unequal-hours calendar is a live
    //! solve path.
    //!
    //! Every expected decider below is hand-derived from the calendar; the
    //! cumulative stage-ends are `[730, 1460, 2190, 2910, 3654, 4374]`.

    use cobre_core::entities::{bus::DeficitSegment, thermal::AnticipatedConfig};
    use cobre_core::scenario::LoadModel;
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig,
    };
    use cobre_core::{
        AnticipatedCommitmentHistory, BoundsCountsSpec, BoundsDefaults, BusStagePenalties,
        ContractBlockBounds, EntityId, HydroBlockBounds, HydroStageBounds, HydroStagePenalties,
        InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingBlockBounds, ResolvedBounds,
        ResolvedPenalties, SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
    };
    use cobre_io::config::TrainingSelection;
    use cobre_io::config::{
        Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig,
        InflowNonNegativityMethod as CfgInflowMethod, ModelingConfig, PolicyConfig,
        RowSelectionConfig, SimulationConfig as IoSimulationConfig, StoppingRuleConfig,
        TrainingConfig, TrainingSolverConfig, UpperBoundEvaluationConfig,
    };
    use cobre_sddp::lead_time::{AnticipatedResolution, LeadTime};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;
    use super::common::build_setup_in_code;
    use super::common::builders::{
        BusSpec, StageSpec, ThermalSpec, make_bus, make_stage, make_thermal,
    };

    /// The d37 unequal-hours monthly calendar (stage totals, hours).
    const D37_DURATIONS: [f64; 6] = [730.0, 730.0, 730.0, 720.0, 744.0, 720.0];
    const N_STAGES: usize = 6;

    /// `LeadStages(2)` on the d37 calendar is the pure index shift
    /// `c(m) = m − 2`: `decider = [None, None, Some(0), Some(1), Some(2), Some(3)]`,
    /// each in-horizon `C(t)` is the singleton `{t + 2}` (t = 4, 5 deliver past the
    /// horizon and are empty), and every `depth` entry is `≤ 2`.
    #[test]
    fn a1c_lead_stages_is_pure_index_shift() {
        let resolution = AnticipatedResolution::resolve(&[LeadTime::Stages(2)], &D37_DURATIONS, 6);
        let point = &resolution.per_plant[0];

        assert_eq!(
            point.decider,
            vec![None, None, Some(0), Some(1), Some(2), Some(3)],
            "LeadStages(2) must be the pure index shift c(m)=m-2",
        );

        for t in 0..N_STAGES {
            if t + 2 < N_STAGES {
                assert_eq!(
                    point.decision_sets[t],
                    vec![t + 2],
                    "in-horizon C({t}) must be the singleton {{{}}}",
                    t + 2,
                );
            } else {
                assert!(
                    point.decision_sets[t].is_empty(),
                    "C({t}) must be empty (delivery t+2 is past the horizon); got {:?}",
                    point.decision_sets[t],
                );
            }
        }

        assert!(
            point.depth.iter().all(|&d| d <= 2),
            "every depth entry must be bounded by the lead 2; got {:?}",
            point.depth,
        );
        assert_eq!(
            resolution.max_fanout, 1,
            "a constant lead is single-decider (|C(t)| <= 1)",
        );
    }

    /// The same `LeadStages(2)` lead resolves identically against a different
    /// 6-stage duration vector (`[672.0; 6]`) — the hour clock is never consulted,
    /// so `decider`, `decision_sets`, and `depth` are byte-for-byte identical.
    #[test]
    fn a1c_lead_stages_ignores_calendar() {
        let on_d37 = AnticipatedResolution::resolve(&[LeadTime::Stages(2)], &D37_DURATIONS, 6);
        let on_uniform = AnticipatedResolution::resolve(&[LeadTime::Stages(2)], &[672.0; 6], 6);

        let a = &on_d37.per_plant[0];
        let b = &on_uniform.per_plant[0];

        assert_eq!(
            a.decider, b.decider,
            "stage-count decider must not depend on the hour clock",
        );
        assert_eq!(
            a.decision_sets, b.decision_sets,
            "stage-count decision_sets must not depend on the hour clock",
        );
        assert_eq!(
            a.depth, b.depth,
            "stage-count depth must not depend on the hour clock",
        );
    }

    /// The physical `LeadTime(1450.0)` mode DOES consult the clock. On the d37
    /// calendar `end_1 − 1450 = 1460 − 1450 = 10`, which lands in stage 0, so the
    /// end-anchored decider decides `m = 1` at stage 0 — where the pure stage-count
    /// shift (`c(1) = 1 − 2`) is still `None`. Full physical decider:
    /// `[None, Some(0), Some(1), Some(1), Some(3), Some(4)]`.
    #[test]
    fn a1c_lead_time_consults_calendar() {
        let physical = AnticipatedResolution::resolve(&[LeadTime::Time(1450.0)], &D37_DURATIONS, 6);
        let stage_count = AnticipatedResolution::resolve(&[LeadTime::Stages(2)], &D37_DURATIONS, 6);

        let phys = &physical.per_plant[0];
        let sc = &stage_count.per_plant[0];

        assert_eq!(
            phys.decider,
            vec![None, Some(0), Some(1), Some(1), Some(3), Some(4)],
            "physical end-anchored decider on the d37 calendar",
        );
        assert_eq!(
            phys.decider[1],
            Some(0),
            "LeadTime consults the clock: m=1 decides at stage 0",
        );
        assert_eq!(
            sc.decider[1], None,
            "LeadStages(2) ignores the clock: c(1)=1-2 is None",
        );
        assert_ne!(
            phys.decider[1], sc.decider[1],
            "the clock-consulted contrast at m=1 (Some(0) vs None)",
        );
    }

    // ── single-decider LeadTime end-to-end on the d37 unequal-hours calendar ──

    const LEAD_TIME_HOURS: f64 = 1440.0;
    const K_MAX: usize = 1;
    const D_LOAD: f64 = 50.0;
    const M_ANT: f64 = 100.0;
    const B_BACK: f64 = 200.0;
    const C_A: f64 = 10.0;
    const C_B: f64 = 100.0;
    const C_DEFICIT: f64 = 1000.0;
    const ITERATIONS: u32 = 12;

    const ANTICIPATED_ID: EntityId = EntityId(2);
    const BACKUP_ID: EntityId = EntityId(3);
    const BUS_ID: EntityId = EntityId(1);

    // SystemBuilder::build() sorts thermals by EntityId ascending: 0 -> id=2
    // (anticipated), 1 -> id=3 (backup).
    const THERMAL_IDX_ANT: usize = 0;
    const THERMAL_IDX_BACKUP: usize = 1;

    /// A single-decider `LeadTime(1440.0)` variant of the d37 topology (1
    /// anticipated thermal, 1 backup thermal, no hydro) on the d37 unequal-hours
    /// calendar. `Time(1440.0)` resolves to `[None, Some(0), Some(1), Some(2),
    /// Some(3), Some(4)]` (`max_fanout = 1`), so it clears the fan-out setup guard;
    /// the single pre-study decider (`decider[0] == None`) fixes the
    /// `past_anticipated_commitments` history length at 1.
    fn build_system() -> cobre_core::System {
        use chrono::NaiveDate;

        let bus = make_bus(
            BUS_ID,
            BusSpec {
                name: "B1".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                deficit_segments: vec![DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: C_DEFICIT,
                }],
                excess_cost: 0.0,
                ..Default::default()
            },
        );

        let thermal_ant = make_thermal(
            ANTICIPATED_ID,
            ThermalSpec {
                name: "T_ant".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: M_ANT,
                cost_per_mwh: C_A,
                anticipated_config: Some(AnticipatedConfig::LeadTime(LEAD_TIME_HOURS)),
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        let thermal_backup = make_thermal(
            BACKUP_ID,
            ThermalSpec {
                name: "T_backup".to_string(),
                operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: BUS_ID,
                min_generation_mw: 0.0,
                max_generation_mw: B_BACK,
                cost_per_mwh: C_B,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
                ..Default::default()
            },
        );

        // Each stage carries one block whose hours are the d37 stage total, so
        // study_stage_durations feeds the [730,730,730,720,744,720] calendar to the
        // point-commitment resolver.
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
                            duration_hours: D37_DURATIONS[i],
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
                mean_mw: D_LOAD,
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
                    filling_min_rate_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                hydro_block: HydroBlockBounds {
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 0.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                    max_diversion_m3s: None,
                },
                thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
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
        );

        // fill_anticipated_columns reads the delivery cell at each decision's own
        // delivery stage; span the full n_stages + k_max axis so every read is
        // well-defined.
        let thermal_axis = N_STAGES + K_MAX;
        for s in 0..thermal_axis {
            *bounds.thermal_bounds_mut(THERMAL_IDX_ANT, s) =
                ThermalStageBounds { cost_per_mwh: C_A };
            *bounds.thermal_block_base_mut(THERMAL_IDX_ANT, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: M_ANT,
            };
            *bounds.thermal_bounds_mut(THERMAL_IDX_BACKUP, s) =
                ThermalStageBounds { cost_per_mwh: C_B };
            *bounds.thermal_block_base_mut(THERMAL_IDX_BACKUP, s) = ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: B_BACK,
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

        // One pre-study delivery stage (decider[0] == None) ⇒ history length 1.
        let initial_conditions = InitialConditions {
            storage: vec![],
            filling_storage: vec![],
            past_anticipated_commitments: vec![AnticipatedCommitmentHistory {
                thermal_id: ANTICIPATED_ID,
                values_mw: vec![0.0],
            }],
            recent_observations: vec![],
            past_defluences: vec![],
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

    fn build_config() -> Config {
        Config {
            schema: None,
            state_space: cobre_io::config::StateSpaceConfig::default(),
            modeling: ModelingConfig {
                inflow_non_negativity: InflowNonNegativityConfig {
                    method: CfgInflowMethod::Penalty,
                },

                cost_scale_factor: None,
            },
            training: TrainingConfig {
                enabled: true,
                tree_seed: Some(42),
                stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                    limit: ITERATIONS,
                }]),
                stopping_mode: cobre_io::config::StoppingMode::Any,
                cut_selection: RowSelectionConfig::default(),
                solver: TrainingSolverConfig::default(),
                parallelism: cobre_io::config::ParallelismConfig::default(),
                scenario_source: None,
                selection: Some(TrainingSelection::Sampled { forward_passes: 1 }),
            },
            upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
            policy: PolicyConfig::default(),
            simulation: IoSimulationConfig::default(),
            exports: ExportsConfig::default(),
            estimation: EstimationConfig::default(),
        }
    }

    /// A single-decider `LeadTime(1440.0)` thermal on the d37 unequal-hours
    /// calendar trains to convergence (`LB == UB`) with no panic — the physical
    /// mode is a live end-to-end path, not merely a resolver unit.
    #[test]
    fn a1c_lead_time_solves_on_unequal_hours() {
        // Guard the fixture's single-decider premise before building the study: a
        // fan-out (max_fanout > 1) would trip the setup guard, not converge.
        let resolution =
            AnticipatedResolution::resolve(&[LeadTime::Time(LEAD_TIME_HOURS)], &D37_DURATIONS, 6);
        assert_eq!(
            resolution.per_plant[0].decider,
            vec![None, Some(0), Some(1), Some(2), Some(3), Some(4)],
            "Time(1440.0) must resolve to the single-decider chain on the d37 calendar",
        );
        assert_eq!(
            resolution.max_fanout, 1,
            "Time(1440.0) must be single-decider (|C(t)| == 1) to clear the fan-out guard",
        );

        let system = build_system();
        let config = build_config();
        let mut setup = build_setup_in_code(system, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");

        assert!(
            outcome.error.is_none(),
            "training error: {:?}",
            outcome.error,
        );

        let result = &outcome.result;
        assert_eq!(
            result.iterations,
            u64::from(ITERATIONS),
            "iteration count mismatch",
        );

        let gap = result.final_gap;
        assert!(
            gap.is_finite() && gap.abs() < 1e-6,
            "final_gap must be ~0 (LB == UB) for a fully deterministic fixture; got {gap}",
        );

        let lb = result.final_lb;
        assert!(
            lb.is_finite() && lb > 0.0,
            "final_lb must be finite and positive; got {lb}",
        );
    }
}
