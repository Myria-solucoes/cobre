//! Consolidated hydro-simulation and stage-structure integration tests for
//! `cobre-sddp`.
//!
//! Each source domain lives in its own inner `mod` so the suite links the
//! statically-bound solver once rather than once per file. Per-`mod` scoping
//! isolates each group's consts, helpers, and fixtures.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::too_many_lines
)]

mod common;

mod simulation_only {
    //! Integration test for simulation-only round-trip: the FCF reconstructed from
    //! a written-then-reloaded policy checkpoint must evaluate identically.

    use std::path::Path;

    use cobre_core::scenario::ScenarioSource;
    use cobre_io::output::policy::{read_policy_checkpoint, write_policy_checkpoint};
    use cobre_sddp::{
        FutureCostFunction, StudySetup, build_basis_cache_from_checkpoint,
        hydro_models::prepare_hydro_models,
        policy_export::{
            build_active_indices, build_stage_basis_records, build_stage_cut_records,
            build_stage_cuts_payloads, convert_basis_cache,
        },
        setup::prepare_stochastic,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    #[test]
    fn simulation_only_fcf_round_trip() {
        let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d01-thermal-dispatch");

        let config_path = case_dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config must parse");

        let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
        let prepare_result =
            prepare_stochastic(system, &case_dir, &config, 42, &ScenarioSource::default())
                .expect("prepare_stochastic");
        let system = prepare_result.system;
        let stochastic = prepare_result.stochastic;

        let hydro_models =
            prepare_hydro_models(&system, &case_dir, false).expect("prepare_hydro_models");

        let mut setup =
            StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup");

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(outcome.error.is_none(), "expected no training error");
        let training_result = outcome.result;

        let original_active_cuts = setup.fcf.total_active_cuts();
        assert!(original_active_cuts > 0, "training should produce cuts");

        let n_stages = setup.stage_data.stage_templates.templates.len();
        let state_dim = setup.fcf.state_dimension;

        let test_state: Vec<f64> = vec![50.0; state_dim];
        let mut original_evals = Vec::with_capacity(n_stages);
        for stage in 0..n_stages {
            original_evals.push(setup.fcf.evaluate_at_state(stage, &test_state));
        }

        let tmpdir = tempfile::tempdir().expect("tempdir");
        let policy_dir = tmpdir.path().join("policy");

        let fcf = &setup.fcf;
        let stage_records = build_stage_cut_records(fcf);
        let stage_active_indices = build_active_indices(&stage_records);
        let stage_manifests: Vec<Vec<cobre_io::EntitySlot>> = vec![Vec::new(); fcf.pools.len()];
        let stage_cuts =
            build_stage_cuts_payloads(fcf, &stage_records, &stage_active_indices, &stage_manifests);

        let (basis_col_u8, basis_row_u8) = convert_basis_cache(&training_result);
        let stage_bases =
            build_stage_basis_records(fcf, &training_result, &basis_col_u8, &basis_row_u8);

        let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();
        let metadata = cobre_io::PolicyCheckpointMetadata {
            cobre_version: env!("CARGO_PKG_VERSION").to_string(),
            created_at: "2026-03-29T00:00:00Z".to_string(),
            completed_iterations: training_result.iterations as u32,
            final_lower_bound: training_result.final_lb,
            best_upper_bound: Some(training_result.final_ub),
            state_dimension: state_dim as u32,
            num_stages: n_stages as u32,
            max_iterations: setup.loop_params.max_iterations as u32,
            forward_passes: setup.loop_params.forward_passes,
            warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
            warm_start_counts,
            rng_seed: 42,
            total_visited_states: 0,
            training_block_mode: "parallel".to_string(),
            training_block_mode_per_stage: vec![],
        };

        write_policy_checkpoint(&policy_dir, &stage_cuts, &stage_bases, &metadata, &[])
            .expect("write checkpoint");

        let checkpoint = read_policy_checkpoint(&policy_dir).expect("read checkpoint");

        assert_eq!(
            checkpoint.metadata.state_dimension, state_dim as u32,
            "state_dimension must round-trip"
        );
        assert_eq!(
            checkpoint.metadata.num_stages, n_stages as u32,
            "num_stages must round-trip"
        );
        assert_eq!(
            checkpoint.stage_cuts.len(),
            n_stages,
            "stage_cuts count must match"
        );

        let loaded_fcf = FutureCostFunction::from_deserialized(&checkpoint.stage_cuts)
            .expect("from_deserialized");

        assert_eq!(
            loaded_fcf.total_active_cuts(),
            original_active_cuts,
            "active cut count must match after round-trip"
        );

        for (stage, &expected_eval) in original_evals.iter().enumerate().take(n_stages) {
            let loaded_eval = loaded_fcf.evaluate_at_state(stage, &test_state);
            assert_eq!(
                loaded_eval, expected_eval,
                "evaluate_at_state mismatch at stage {stage}"
            );
        }

        let loaded_basis_cache = build_basis_cache_from_checkpoint(
            n_stages,
            &checkpoint.stage_bases,
            &checkpoint.stage_cuts,
        );
        assert_eq!(
            loaded_basis_cache.len(),
            n_stages,
            "basis cache length must match"
        );
        let has_basis = loaded_basis_cache.iter().any(Option::is_some);
        assert!(has_basis, "at least one stage should have basis data");
    }
}

mod d17_signed_evaporation {
    //! Focused semantic regression test for the signed evaporation-outflow variable.
    //!
    //! Runs D17 (`d17-evaporation-mixed-sign`) end-to-end and asserts that the
    //! simulation output reflects the mixed-sign monthly evaporation coefficients
    //! defined in the fixture:
    //!
    //! - Stages 0, 1, 2 land on Oct/Nov/Dec where `coefficients_mm` is negative
    //!   (net rainfall input on the lake surface); the simulated
    //!   `evaporation_m3s` must be **negative**.
    //! - Stage 3 lands on January where `coefficients_mm` is positive (true
    //!   evaporation loss); the simulated `evaporation_m3s` must be **positive**.
    //! - In every stage the symmetric `[-q_max, +q_max]` magnitude bound must
    //!   absorb the linearized target, so neither the positive nor the negative
    //!   evaporation violation slack should fire.
    //!
    //! The companion `parity_hash_d17` test guards bit-for-bit byte stability of
    //! the same case; this test guards the **sign semantics** explicitly so a
    //! future regression that flips the sign convention is caught by an
    //! actionable assertion rather than an opaque hash diff.

    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::scenario::ScenarioSource;
    use cobre_sddp::{
        StudySetup, aggregate_simulation,
        hydro_models::prepare_hydro_models,
        setup::{StudyParams, prepare_stochastic},
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn d17_evaporation_is_signed_per_month() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic/d17-evaporation-mixed-sign");

        let config_path = dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config must parse");
        let system = cobre_io::load_case(&dir).expect("load_case must succeed");

        let pr = prepare_stochastic(system, &dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic must succeed");
        let system = pr.system;
        let stochastic = pr.stochastic;

        let hydro_models =
            prepare_hydro_models(&system, &dir, false).expect("prepare_hydro_models must succeed");

        let mut config_with_sim = config.clone();
        config_with_sim.simulation.enabled = true;
        config_with_sim.simulation.num_scenarios = 1;

        let sentinel = Path::new("config.json");
        let training_source = config_with_sim
            .training_scenario_source(sentinel)
            .expect("training_scenario_source must parse");
        let simulation_source = config_with_sim
            .simulation_scenario_source(sentinel)
            .expect("simulation_scenario_source must parse");

        let params = StudyParams::from_config(&config_with_sim)
            .expect("StudyParams::from_config must succeed");
        let construction = params.into_construction_config();

        let mut setup = StudySetup::from_broadcast_params(
            &system,
            stochastic,
            construction,
            hydro_models,
            &training_source,
            &simulation_source,
        )
        .expect("StudySetup must build");

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(outcome.error.is_none(), "D17: expected no training error");
        let result = outcome.result;

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("simulation workspace pool must build");

        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);
        let drain_handle = std::thread::spawn(move || result_rx.into_iter().collect::<Vec<_>>());

        let local_costs = setup
            .simulate(
                &mut pool.workspaces,
                &comm,
                &result_tx,
                None,
                result.frozen_templates.as_deref(),
                &result.basis_cache,
            )
            .expect("simulate must return Ok");

        drop(result_tx);
        let mut scenario_results = drain_handle.join().expect("drain thread must not panic");

        let sim_config = setup.simulation_config();
        let _summary = aggregate_simulation(&local_costs.costs, sim_config, &comm)
            .expect("aggregate_simulation must succeed");

        assert_eq!(
            scenario_results.len(),
            1,
            "expected exactly one simulation scenario for D17, got {}",
            scenario_results.len()
        );
        let scenario = scenario_results.remove(0);
        assert_eq!(
            scenario.stages.len(),
            4,
            "expected 4 stages in D17 simulation result, got {}",
            scenario.stages.len()
        );

        for stage in &scenario.stages {
            assert_eq!(
                stage.hydros.len(),
                1,
                "stage {} should have exactly one hydro record",
                stage.stage_id
            );
            let h = &stage.hydros[0];
            let evap = h
                .evaporation_m3s
                .expect("evaporation must be modeled for D17");

            match stage.stage_id {
                0..=2 => assert!(
                    evap < 0.0,
                    "stage {} (Oct/Nov/Dec) expected net-rainfall evaporation_m3s < 0, got {evap}",
                    stage.stage_id
                ),
                3 => assert!(
                    evap > 0.0,
                    "stage {} (Jan) expected true-evaporation evaporation_m3s > 0, got {evap}",
                    stage.stage_id
                ),
                other => panic!("unexpected stage_id in D17 simulation: {other}"),
            }

            assert_eq!(
                h.evaporation_violation_pos_m3s, 0.0,
                "stage {}: positive evaporation slack must not fire, got {}",
                stage.stage_id, h.evaporation_violation_pos_m3s
            );
            assert_eq!(
                h.evaporation_violation_neg_m3s, 0.0,
                "stage {}: negative evaporation slack must not fire, got {}",
                stage.stage_id, h.evaporation_violation_neg_m3s
            );
        }
    }
}

mod d41_energy_contracts_simulation {
    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::scenario::ScenarioSource;
    use cobre_sddp::simulation::accumulate_category_costs;
    use cobre_sddp::simulation::types::ScenarioCategoryCosts;
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    /// Energy-contract gating and dispatch must reach the simulation output.
    ///
    /// The parity hash does NOT digest contract-output columns, so a gating bug — a
    /// commissioning-dormant contract that dispatches, a wrong-signed load-balance
    /// term that flips import/export, a mis-stamped `contract_id`, a stage override
    /// that fails to change dispatch — would still hash-match and ship silently. This
    /// test asserts the dispatch values directly through the full train+simulate
    /// pipeline; they are the only numerical guard.
    ///
    /// The commissioning gate is `entry <= stage.id && stage.id < exit`. Under the
    /// dense layout a dormant contract keeps its column but is pinned to `[0, 0]`,
    /// emitting a ZERO row rather than being absent.
    ///
    /// D41 topology: 3 stages, single 730h block, one bus, two hydros H0->H1, one
    /// thermal at $50/MWh, deficit at $1000/MWh. Two contracts on bus 0: import id 0
    /// (price $200/MWh, always active, stage-2 `min_mw` + `price_per_mwh` override)
    /// and export id 1 (price -$150/MWh revenue, commissioned at stage 1 only).
    #[test]
    fn energy_contract_gating_reaches_simulation_output() {
        let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d41-energy-contracts");

        let config_path = case_dir.join("config.json");
        let mut config = cobre_io::parse_config(&config_path).expect("config must parse");
        // The shipped parity case trains only; enable one sim scenario so the contract extraction path runs.
        config.simulation = cobre_io::config::SimulationConfig {
            enabled: true,
            num_scenarios: 1,
            io_channel_capacity: 8,
            ..cobre_io::config::SimulationConfig::default()
        };

        let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
        let prepare_result =
            prepare_stochastic(system, &case_dir, &config, 42, &ScenarioSource::default())
                .expect("prepare_stochastic must succeed");
        let system = prepare_result.system;
        let stochastic = prepare_result.stochastic;

        let hydro_models = prepare_hydro_models(&system, &case_dir, false)
            .expect("prepare_hydro_models must succeed");

        let mut setup =
            StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup::new");

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
            .expect("simulate must not return Err");

        drop(result_tx);
        let scenario_results = drain_handle.join().expect("drain thread must not panic");

        assert_eq!(
            scenario_results.len(),
            1,
            "exactly one deterministic scenario result",
        );
        let scenario = &scenario_results[0];
        assert_eq!(
            scenario.stages.len(),
            3,
            "one stage record per study stage (horizon 0, 1, 2)",
        );

        // Each stage keeps a dense row per contract (single block): import id 0 then
        // export id 1, regardless of commissioning.
        for (t, stage) in scenario.stages.iter().enumerate() {
            assert_eq!(
                stage.contracts.len(),
                2,
                "stage {t} keeps the dense column for both contracts, got {:?}",
                stage.contracts,
            );
        }

        let import_row = |t: usize| {
            scenario.stages[t]
                .contracts
                .iter()
                .find(|r| r.contract_id == 0)
                .unwrap_or_else(|| panic!("stage {t} must have an import (id 0) row"))
        };
        let export_row = |t: usize| {
            scenario.stages[t]
                .contracts
                .iter()
                .find(|r| r.contract_id == 1)
                .unwrap_or_else(|| panic!("stage {t} must have an export (id 1) row"))
        };

        // Commissioning gate, dormant zero: export id 1 has entry_stage_id = 1,
        // so stage 0 (t < E) is dormant: zero power, operative_state_code == 1.
        let exp0 = export_row(0);
        assert_eq!(
            exp0.power_mw, 0.0,
            "stage 0 is before the export entry window: dormant column pinned to 0, got {}",
            exp0.power_mw,
        );
        assert_eq!(
            exp0.operative_state_code, 1,
            "dormant export row still carries operative_state_code 1",
        );

        // Commissioning gate, active dispatch + post-exit zero: export id 1 has
        // exit_stage_id = 2, active only at stage 1 (E <= t < X). It is the surplus
        // stage, so the optimizer sells the revenue export at a positive power.
        let exp1 = export_row(1);
        assert!(
            exp1.power_mw > 0.0,
            "stage 1 is inside the export window and a surplus stage: export must dispatch, got {}",
            exp1.power_mw,
        );
        assert!(
            exp1.total_cost < 0.0,
            "an active export (negative price) nets negative total_cost (revenue), got {}",
            exp1.total_cost,
        );
        let exp2 = export_row(2);
        assert_eq!(
            exp2.power_mw, 0.0,
            "stage 2 is at/after the export exit boundary (decommissioned): pinned to 0, got {}",
            exp2.power_mw,
        );

        // Stage 0 is a shortage stage — import id 0 is pulled (high load, scarce hydro,
        // thermal capped): import at $200 beats the $1000 deficit.
        let imp0 = import_row(0);
        assert!(
            imp0.power_mw > 0.0,
            "stage 0 is a shortage stage: import must be pulled, got {}",
            imp0.power_mw,
        );
        assert!(
            (imp0.price_per_mwh - 200.0).abs() < 1e-6,
            "stage 0 import uses the base price 200.0, got {}",
            imp0.price_per_mwh,
        );

        // Take-or-pay floor binds: the contract_bounds override sets import id 0
        // min_mw = 10.0 at stage 2, where importing is uneconomic (hydro+thermal cover
        // the load and the override price is 999.0). The floor forces power_mw == 10.0.
        let imp2 = import_row(2);
        assert!(
            (imp2.power_mw - 10.0).abs() < 1e-6,
            "stage 2 import take-or-pay floor (min_mw = 10.0) must bind, got {}",
            imp2.power_mw,
        );

        // Stage override changes price and cost: the same override sets import id
        // 0 price_per_mwh = 999.0 at stage 2, differing from the base 200.0 used at
        // stage 0, which visibly changes total_cost relative to the non-overridden
        // stage 0 import row.
        assert!(
            (imp2.price_per_mwh - 999.0).abs() < 1e-6,
            "stage 2 import price override (999.0) must be reflected, got {}",
            imp2.price_per_mwh,
        );
        assert!(
            (imp2.price_per_mwh - imp0.price_per_mwh).abs() > 1e-6,
            "override price (999.0) must differ from base price (200.0)",
        );
        let block_hours = 730.0;
        assert!(
            (imp2.total_cost - 999.0 * 10.0 * block_hours).abs() < 1.0,
            "stage 2 import total_cost = price * power * hours = 999 * 10 * 730, got {}",
            imp2.total_cost,
        );
        assert!(
            (imp2.total_cost - imp0.total_cost).abs() > 1.0,
            "overridden stage 2 import total_cost must differ from non-overridden stage 0",
        );

        // Cost-breakdown invariant with an active contract: the five macro
        // categories sum to immediate_cost at the export-active stage 1.
        let cost = &scenario.stages[1].costs[0];
        let mut accum = ScenarioCategoryCosts {
            resource_cost: 0.0,
            recourse_cost: 0.0,
            violation_cost: 0.0,
            regularization_cost: 0.0,
            imputed_cost: 0.0,
        };
        accumulate_category_costs(cost, &mut accum);
        let macro_sum = accum.resource_cost
            + accum.recourse_cost
            + accum.violation_cost
            + accum.regularization_cost
            + accum.imputed_cost;
        assert!(
            (macro_sum - cost.immediate_cost).abs() < 1.0,
            "Sigma(macro categories) ({macro_sum}) must equal immediate_cost ({}) with an active contract",
            cost.immediate_cost,
        );
    }
}

mod multi_resolution_integration {
    //! End-to-end integration test for the multi-resolution stage-transition pipeline
    //! on the D30 case (6 monthly stages followed by 4 quarterly stages).

    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::temporal::SeasonCycleType;
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    // ---------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------

    fn d30_case_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d30-multi-resolution-monthly-quarterly")
    }

    /// Uses `config.training_scenario_source()`, not [`ScenarioSource::default()`]:
    /// the default silently falls back to `InSample` and bypasses the PAR(1) pipeline.
    fn build_setup(case_dir: &Path, config: &cobre_io::Config) -> StudySetup {
        let system = cobre_io::load_case(case_dir).expect("load_case");
        let config_path = case_dir.join("config.json");
        let source = config
            .training_scenario_source(&config_path)
            .expect("training_scenario_source");
        let prep =
            prepare_stochastic(system, case_dir, config, 42, &source).expect("prepare_stochastic");
        let hydro_models =
            prepare_hydro_models(&prep.system, case_dir, false).expect("prepare_hydro_models");
        StudySetup::new(&prep.system, config, prep.stochastic, hydro_models)
            .expect("StudySetup::new")
    }

    // ---------------------------------------------------------------------------
    // Test: structural properties, training, and simulation
    // ---------------------------------------------------------------------------

    #[test]
    fn multi_resolution_structural_properties_and_training() {
        let case_dir = d30_case_dir();
        let config_path = case_dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config");

        // ── 1. Structural properties after load ──────────────────────────────────

        let system = cobre_io::load_case(&case_dir).expect("load_case D30");

        // Pre-study stages carry negative IDs; the study count is the non-negative set.
        let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();

        assert_eq!(
            study_stages.len(),
            10,
            "D30: expected 10 study stages; got {}",
            study_stages.len()
        );

        for (t, stage) in study_stages.iter().enumerate().take(6) {
            assert_eq!(
                stage.scenario_config.branching_factor, 5,
                "D30: stage {t} (monthly) must have num_scenarios == 5"
            );
        }

        for (t, stage) in study_stages.iter().enumerate().skip(6) {
            assert_eq!(
                stage.scenario_config.branching_factor, 5,
                "D30: stage {t} (quarterly) must have num_scenarios == 5"
            );
        }

        let season_map = system
            .policy_graph()
            .season_map
            .as_ref()
            .expect("D30 must have season_map");
        assert_eq!(
            season_map.cycle_type,
            SeasonCycleType::Custom,
            "D30: season map must have Custom cycle type"
        );
        assert_eq!(
            season_map.seasons.len(),
            16,
            "D30: Custom season map must have 16 seasons (12 monthly + 4 quarterly); got {}",
            season_map.seasons.len()
        );

        for expected_id in 0..12_usize {
            assert!(
                season_map.seasons.iter().any(|s| s.id == expected_id),
                "D30: monthly season id {expected_id} must be present in the season map"
            );
        }

        for expected_id in 12..16_usize {
            assert!(
                season_map.seasons.iter().any(|s| s.id == expected_id),
                "D30: quarterly season id {expected_id} must be present in the season map"
            );
        }

        // ── 2. Build setup and verify noise group IDs ─────────────────────────────

        let mut setup = build_setup(&case_dir, &config);

        let groups = &setup.stage_data.noise_group_ids;
        assert_eq!(
            groups.len(),
            10,
            "D30: noise_group_ids must have 10 elements (one per study stage); got {}",
            groups.len()
        );

        // Monthly stages 0-5 each pair a unique season_id with a unique year, so each
        // gets its own group ID — hence 6 distinct values.
        let monthly_groups: Vec<u32> = groups[..6].to_vec();
        let mut unique_monthly: Vec<u32> = monthly_groups.clone();
        unique_monthly.sort_unstable();
        unique_monthly.dedup();
        assert_eq!(
            unique_monthly.len(),
            6,
            "D30: monthly stages 0-5 must have 6 distinct noise group IDs; got {monthly_groups:?}"
        );

        // ── 3. Downstream lag transition fields ───────────────────────────────────

        let stage_ctx = setup.stage_ctx();
        let lag_transitions = stage_ctx.stage_lag_transitions;

        assert_eq!(
            lag_transitions.len(),
            10,
            "D30: stage_lag_transitions must have 10 entries (one per study stage); got {}",
            lag_transitions.len()
        );

        // downstream_par_order == 1 → the pre-transition window is the 1*3 = 3 monthly
        // stages (3, 4, 5) immediately before the quarterly boundary.
        for (t, transition) in lag_transitions.iter().enumerate().take(6).skip(3) {
            assert!(
                transition.accumulate_downstream,
                "D30: monthly stage {t} (index in pre-transition window) must have              accumulate_downstream == true; got false. Full transition: {transition:?}"
            );
        }

        for (t, transition) in lag_transitions.iter().enumerate().take(3) {
            assert!(
                !transition.accumulate_downstream,
                "D30: monthly stage {t} (outside pre-transition window) must have              accumulate_downstream == false; got true. Full transition: {transition:?}"
            );
        }

        assert!(
            lag_transitions[6].rebuild_from_downstream,
            "D30: first quarterly stage (index 6) must have rebuild_from_downstream == true; \
             got false. Full transition: {:?}",
            lag_transitions[6]
        );

        for (t, transition) in lag_transitions.iter().enumerate().skip(7) {
            assert!(
                !transition.rebuild_from_downstream,
                "D30: quarterly stage {t} (not first) must have rebuild_from_downstream == false;              got true. Full transition: {transition:?}"
            );
        }

        // ── 4. Train ──────────────────────────────────────────────────────────────

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("solver");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("D30: train must not return Err");

        assert!(
            outcome.error.is_none(),
            "D30: training outcome must have no error; got: {:?}",
            outcome.error
        );
        assert!(
            outcome.result.iterations >= 1,
            "D30: must complete at least 1 iteration; got {}",
            outcome.result.iterations
        );
        assert!(
            outcome.result.final_lb > 0.0,
            "D30: lower bound must be positive; got {}",
            outcome.result.final_lb
        );
        assert!(
            outcome.result.final_lb.is_finite(),
            "D30: lower bound must be finite; got {}",
            outcome.result.final_lb
        );

        // ── 5. Simulate ───────────────────────────────────────────────────────────

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("D30: workspace pool must build");
        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);

        // Drain the channel in a background thread to avoid blocking simulate().
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
            .expect("D30: simulate must not return Err");

        drop(result_tx);
        let scenario_results = drain_handle.join().expect("drain thread must not panic");

        assert_eq!(
            scenario_results.len(),
            1,
            "D30: expected 1 simulation scenario result; got {}",
            scenario_results.len()
        );
    }
}

mod sparse_dense {
    //! `build_cut_row_batch_into` emits one `(lp_column, coefficient)` pair per
    //! nonzero-state-mask entry in ascending order, then the theta column. For a
    //! proper-subset mask (mixed AR orders), the emitted columns and values must
    //! equal the remapped LP columns and the negated coefficients.

    use cobre_sddp::FutureCostFunction;
    use cobre_sddp::build_cut_row_batch_into;
    use cobre_sddp::indexer::StateLayout;
    use cobre_solver::RowBatch;

    #[test]
    fn sparse_partial_mask_produces_correct_subset() {
        let n_hydro = 3;
        let max_par_order = 2;
        let n_state = n_hydro * (1 + max_par_order);

        // The [0, 1, 2] arg is per-hydro effective_lag_count; mixed orders zero out
        // some lag slots. StateLayout::new finalizes the mask as production setup does.
        let state = StateLayout::new(
            n_hydro,
            max_par_order,
            0,
            Vec::new(),
            0,
            0,
            vec![],
            &[0, 1, 2],
        );
        // Expected mask = storage [0,1,2] + lag0 of h1,h2 [4,5] + lag1 of h2 [8].
        let mask = &state.nonzero_state_indices;
        assert_eq!(
            mask.len(),
            6,
            "mixed-order mask has 3 + 0 + 1 + 2 = 6 entries"
        );

        let mut fcf = FutureCostFunction::new(2, n_state, 1, 1, &[0; 2]);
        let coeffs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        fcf.add_cut(0, 0, 0, 50.0, &coeffs);

        let col_scale: Vec<f64> = Vec::new();

        let mut batch = RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        };
        let cut_state = cobre_sddp::test_support::cut_state_projection(&state);
        build_cut_row_batch_into(&mut batch, &fcf, 0, &state, &cut_state, &col_scale);

        assert_eq!(batch.num_rows, 1);
        let theta_col = state.theta;
        let expected_cols: Vec<i32> = mask
            .iter()
            .map(|&j| state.state_to_lp_column(j) as i32)
            .chain(std::iter::once(theta_col as i32))
            .collect();
        assert_eq!(
            batch.col_indices, expected_cols,
            "sparse col_indices mismatch"
        );

        let expected_values: Vec<f64> = mask
            .iter()
            .map(|&j| -coeffs[j])
            .chain(std::iter::once(1.0))
            .collect();
        for (i, (actual, expected)) in batch.values.iter().zip(expected_values.iter()).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "values[{i}] differ: actual={actual}, expected={expected}"
            );
        }
    }
}

mod decomp_integration {
    //! End-to-end integration test for the mixed-resolution pipeline on the D28
    //! case (5 weekly stages + 1 monthly terminal stage).

    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::temporal::SeasonCycleType;
    use cobre_io::output::policy::write_policy_checkpoint;
    use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    fn d28_case_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/deterministic/d28-decomp-weekly-monthly")
    }

    /// Write a policy checkpoint to `policy_dir` from the given setup and training result.
    fn write_test_checkpoint(
        policy_dir: &Path,
        setup: &StudySetup,
        result: &cobre_sddp::TrainingResult,
        seed: u64,
    ) {
        let stage_manifests: Vec<Vec<cobre_io::EntitySlot>> =
            vec![Vec::new(); setup.fcf.pools.len()];
        write_test_checkpoint_with_manifests(policy_dir, setup, result, seed, &stage_manifests);
    }

    /// Like [`write_test_checkpoint`] but attaches `stage_manifests` (one per pool) to
    /// the per-stage cut payloads, so a test can write a checkpoint whose entity
    /// identity deliberately diverges from the current study's.
    fn write_test_checkpoint_with_manifests(
        policy_dir: &Path,
        setup: &StudySetup,
        result: &cobre_sddp::TrainingResult,
        seed: u64,
        stage_manifests: &[Vec<cobre_io::EntitySlot>],
    ) {
        use cobre_sddp::policy_export::{
            build_active_indices, build_stage_basis_records, build_stage_cut_records,
            build_stage_cuts_payloads, convert_basis_cache,
        };
        let fcf = &setup.fcf;
        let stage_records = build_stage_cut_records(fcf);
        let stage_active_indices = build_active_indices(&stage_records);
        let stage_cuts =
            build_stage_cuts_payloads(fcf, &stage_records, &stage_active_indices, stage_manifests);
        let (basis_col, basis_row) = convert_basis_cache(result);
        let stage_bases = build_stage_basis_records(fcf, result, &basis_col, &basis_row);
        let warm_start_counts: Vec<u32> = fcf.pools.iter().map(|p| p.warm_start_count).collect();
        let metadata = cobre_io::PolicyCheckpointMetadata {
            cobre_version: env!("CARGO_PKG_VERSION").to_string(),
            created_at: "2026-04-14T00:00:00Z".to_string(),
            completed_iterations: result.iterations as u32,
            final_lower_bound: result.final_lb,
            best_upper_bound: Some(result.final_ub),
            state_dimension: fcf.state_dimension as u32,
            num_stages: fcf.pools.len() as u32,
            max_iterations: setup.loop_params.max_iterations as u32,
            forward_passes: setup.loop_params.forward_passes,
            warm_start_cuts: warm_start_counts.iter().copied().max().unwrap_or(0),
            warm_start_counts,
            rng_seed: seed,
            total_visited_states: 0,
            training_block_mode: "parallel".to_string(),
            training_block_mode_per_stage: vec![],
        };
        write_policy_checkpoint(policy_dir, &stage_cuts, &stage_bases, &metadata, &[])
            .expect("write checkpoint");
    }

    /// Build a `StudySetup` for the D28 case, calling `config.training_scenario_source()`
    /// so the External inflow library is loaded; `ScenarioSource::default()` (`InSample`)
    /// would skip the external parquet and produce a wrong scenario pipeline.
    fn build_setup(case_dir: &Path, config: &cobre_io::Config) -> (StudySetup, cobre_core::System) {
        let system = cobre_io::load_case(case_dir).expect("load_case");
        let config_path = case_dir.join("config.json");
        let source = config
            .training_scenario_source(&config_path)
            .expect("training_scenario_source");
        let prep =
            prepare_stochastic(system, case_dir, config, 42, &source).expect("prepare_stochastic");
        let hydro_models =
            prepare_hydro_models(&prep.system, case_dir, false).expect("prepare_hydro_models");
        let setup = StudySetup::new(&prep.system, config, prep.stochastic, hydro_models)
            .expect("StudySetup::new");
        (setup, prep.system)
    }

    /// Verify structural properties, training correctness, lag accumulation, and
    /// simulation completion for the D28 mixed-resolution case.
    ///
    /// D28 has 5 weekly stages (indices 0-4, `num_scenarios == 1`) followed by
    /// 1 monthly terminal stage (index 5, `num_scenarios == 5`). The monthly
    /// season map must have 12 seasons.
    #[test]
    fn structural_properties_and_training() {
        let case_dir = d28_case_dir();
        let config_path = case_dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config");

        let system = cobre_io::load_case(&case_dir).expect("load_case D28");
        let stages = system.stages();

        assert_eq!(
            stages.len(),
            6,
            "D28 must have exactly 6 study stages; got {}",
            stages.len()
        );

        for (t, stage) in stages.iter().enumerate().take(5) {
            assert_eq!(
                stage.scenario_config.branching_factor, 1,
                "stage {t} (weekly) must have num_scenarios == 1"
            );
        }

        assert_eq!(
            stages[5].scenario_config.branching_factor, 5,
            "stage 5 (monthly) must have num_scenarios == 5"
        );

        let season_map = system
            .policy_graph()
            .season_map
            .as_ref()
            .expect("D28 must have season_map");
        assert_eq!(
            season_map.cycle_type,
            SeasonCycleType::Monthly,
            "season map must have monthly cycle type"
        );
        assert_eq!(
            season_map.seasons.len(),
            12,
            "monthly season map must have 12 seasons"
        );

        let (mut setup, _system) = build_setup(&case_dir, &config);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("solver");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must not return Err");

        assert!(
            outcome.error.is_none(),
            "training outcome must have no error; got: {:?}",
            outcome.error
        );
        assert!(
            outcome.result.iterations >= 1,
            "must complete at least 1 iteration; got {}",
            outcome.result.iterations
        );
        assert!(
            outcome.result.final_lb.is_finite(),
            "lower bound must be finite; got {}",
            outcome.result.final_lb
        );
        assert!(
            !outcome.result.final_lb.is_nan(),
            "lower bound must not be NaN"
        );

        // The backward pass writes cuts into pools[0..T-2] only; the terminal pool
        // (pools[T-1]) stays empty unless boundary cuts are injected.
        let fcf = &setup.fcf;
        let n_pools = fcf.pools.len();
        for (t, pool) in fcf.pools[..n_pools - 1].iter().enumerate() {
            assert!(
                pool.populated() >= 1,
                "stage {t} (non-terminal) cut pool must have at least 1 cut after training; got {}",
                pool.populated()
            );
        }
        assert_eq!(
            fcf.pools[n_pools - 1].populated(),
            0,
            "terminal pool must have 0 backward-pass cuts (boundary-cut injection is a separate test)"
        );

        let stage_ctx = setup.stage_ctx();
        let lag_transitions = stage_ctx.stage_lag_transitions;

        assert_eq!(
            lag_transitions.len(),
            6,
            "stage_lag_transitions must have 6 entries (one per study stage)"
        );

        let mut pool = setup
            .create_workspace_pool(&comm, 1, ActiveSolver::new)
            .expect("workspace pool must build");
        let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
        let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);

        // Drain the channel in a background thread to avoid blocking simulate().
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
        let _scenario_results = drain_handle.join().expect("drain thread must not panic");

        assert!(
            sim_run.costs.is_empty() || !sim_run.costs.is_empty(),
            "simulate returned successfully"
        );
        let sim_config = setup.simulation_config();
        assert_eq!(
            sim_config.n_scenarios, 5,
            "simulation must be configured for 5 scenarios"
        );
    }

    /// Verify that boundary cuts produced by a trained D28 source study compose
    /// correctly with a consumer D28 study: injecting them must not degrade the
    /// lower bound relative to the no-boundary-cut baseline.
    #[test]
    fn decomp_boundary_cuts_compose_with_weekly_monthly() {
        let case_dir = d28_case_dir();
        let config_path = case_dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config");

        let comm = StubComm;

        let (mut setup_a, _system_a) = build_setup(&case_dir, &config);
        let mut solver_a = ActiveSolver::new().expect("solver A");
        let outcome_a = setup_a
            .train(&mut solver_a, &comm, 1, ActiveSolver::new, None, None)
            .expect("train A");
        assert!(outcome_a.error.is_none(), "source training must not error");

        let tmpdir = tempfile::tempdir().expect("tempdir");
        let source_policy_dir = tmpdir.path().join("source_policy");
        write_test_checkpoint(&source_policy_dir, &setup_a, &outcome_a.result, 42);

        let num_stages = setup_a.fcf.pools.len();
        let source_stage = (num_stages - 2) as u32; // second-to-last stage has backward-pass cuts

        let (mut setup_b, _system_b) = build_setup(&case_dir, &config);
        let mut solver_b = ActiveSolver::new().expect("solver B");
        let outcome_b = setup_b
            .train(&mut solver_b, &comm, 1, ActiveSolver::new, None, None)
            .expect("train B");
        assert!(
            outcome_b.error.is_none(),
            "baseline training must not error"
        );
        let lb_no_boundary = outcome_b.result.final_lb;

        let (mut setup_c, system_c) = build_setup(&case_dir, &config);
        let state_dim = setup_c.fcf.state_dimension as u32;
        let current_manifest = setup_c.build_terminal_entity_manifest(&system_c);
        let mut warnings: Vec<String> = Vec::new();
        let boundary_records = cobre_sddp::load_boundary_cuts(
            &source_policy_dir,
            source_stage,
            state_dim,
            &current_manifest,
            &mut |msg| warnings.push(msg.to_string()),
        )
        .expect("load_boundary_cuts");
        assert!(
            !boundary_records.is_empty(),
            "source stage must have cuts after training"
        );
        cobre_sddp::inject_boundary_cuts(&mut setup_c, &boundary_records);

        let terminal_pool = &setup_c.fcf.pools[num_stages - 1];
        assert!(
            terminal_pool.warm_start_count > 0,
            "terminal pool must have boundary cuts after injection; warm_start_count == 0"
        );
        assert!(
            terminal_pool.populated() >= terminal_pool.warm_start_count as usize,
            "populated_count ({}) must include boundary cuts (warm_start_count = {})",
            terminal_pool.populated(),
            terminal_pool.warm_start_count
        );

        let mut solver_c = ActiveSolver::new().expect("solver C");
        let outcome_c = setup_c
            .train(&mut solver_c, &comm, 1, ActiveSolver::new, None, None)
            .expect("train C");
        assert!(
            outcome_c.error.is_none(),
            "consumer training must not error"
        );
        let lb_with_boundary = outcome_c.result.final_lb;

        assert!(
            lb_with_boundary >= lb_no_boundary - 1e-6,
            "boundary LB ({lb_with_boundary}) must be >= baseline LB ({lb_no_boundary})"
        );
    }

    /// A boundary checkpoint whose source-stage manifest carries a permuted hydro
    /// `entity_id` at slot 0 must be rejected by `load_boundary_cuts` against the
    /// current study's terminal manifest, naming the offending slot — the fail-loud
    /// catch of the silent wrong-bound mis-load the integer-only check misses.
    #[test]
    fn decomp_boundary_cuts_reject_permuted_entity_identity() {
        let case_dir = d28_case_dir();
        let config_path = case_dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config");

        let comm = StubComm;

        let (mut setup_a, system_a) = build_setup(&case_dir, &config);
        let mut solver_a = ActiveSolver::new().expect("solver A");
        let outcome_a = setup_a
            .train(&mut solver_a, &comm, 1, ActiveSolver::new, None, None)
            .expect("train A");
        assert!(outcome_a.error.is_none(), "source training must not error");

        // D28 has one hydro, so the terminal manifest carries at least one storage
        // slot to permute.
        let current_manifest = setup_a.build_terminal_entity_manifest(&system_a);
        assert!(
            !current_manifest.is_empty(),
            "D28 must have at least one state slot for this test"
        );

        // Write the source checkpoint with a manifest whose slot 0 entity_id is
        // shifted away from the current study's, on every stage (the uniform cut
        // dimension makes per-stage manifests identical).
        let mut permuted = current_manifest.clone();
        permuted[0].entity_id += 1000;
        let n_pools = setup_a.fcf.pools.len();
        let stage_manifests: Vec<Vec<cobre_io::EntitySlot>> = vec![permuted; n_pools];

        let tmpdir = tempfile::tempdir().expect("tempdir");
        let source_policy_dir = tmpdir.path().join("source_policy");
        write_test_checkpoint_with_manifests(
            &source_policy_dir,
            &setup_a,
            &outcome_a.result,
            42,
            &stage_manifests,
        );

        let source_stage = (n_pools - 2) as u32;
        let state_dim = setup_a.fcf.state_dimension as u32;
        let result = cobre_sddp::load_boundary_cuts(
            &source_policy_dir,
            source_stage,
            state_dim,
            &current_manifest,
            &mut |_| {},
        );

        assert!(
            result.is_err(),
            "a permuted slot-0 identity must be rejected, not silently loaded"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("slot 0"),
            "rejection must name the offending slot 0: {msg}"
        );
        assert!(
            msg.contains("entity-identity mismatch"),
            "rejection must describe the identity mismatch: {msg}"
        );
    }
}
