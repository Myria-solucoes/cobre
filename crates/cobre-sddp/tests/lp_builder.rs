//! Consolidated LP-builder / indexer / stage-geometry integration tests for
//! `cobre-sddp`.
//!
//! Groups the non-uniform-block extraction backstop, the indexer-slim migration
//! rejection gate, the operational-start-date fixture-ordering guard, and the
//! policy entity-manifest coverage into one binary so the statically-linked solver
//! links once rather than once per file. The extraction backstop is HiGHS-only, so
//! it carries a per-`mod` `#[cfg(feature = "highs")]`; the other three mods compile
//! on every backend. Only the extraction and policy mods reach the shared
//! `tests/common` harness.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

mod common;

#[cfg(feature = "highs")]
mod extraction_nonuniform_block_bases {
    //! Non-uniform-block extraction-correctness assertions.
    //!
    //! These tests pin the simulation read-path against the equipment-base /
    //! cost-range bug that surfaces only when a stage's block count differs from
    //! stage 0's. Both D33 (`[1, 3, 2]`, no anticipated thermals) and D34
    //! (`[1, 3, 2]` plus an anticipated thermal) declare such a schedule, so any
    //! family read with the global stage-0 base/length misreads the interior stages.
    //!
    //! Two assertions, each FAILS against the pre-fix stage-0-base/length read and
    //! PASSES once extraction resolves the per-stage `StageGeometry`:
    //!
    //! 1. **Per-block equipment shape** — at every stage the simulation must emit one
    //!    hydro record per (block, hydro) pair, i.e. exactly `n_blks(stage)` records
    //!    per hydro. A wrong per-stage stride/base does not change the record *count*
    //!    but does change which column each record reads; this assertion pins the
    //!    count as a coarse structural guard and the reconciliation below pins the
    //!    *values*.
    //! 2. **Cost-breakdown reconciliation** — `Σ(cost categories)` must equal
    //!    `immediate_cost` at every stage, including the non-uniform interior stages.
    //!    Pre-fix, `compute_cost_result` sums the stage-0 ranges (wrong base AND
    //!    length) for the interior stages, so the breakdown no longer reconciles to
    //!    the solved objective; post-fix it sums the stage-correct ranges and
    //!    reconciles exactly. D33/D34 declare no generic constraints and no NCS, so
    //!    every cost category is an objective·primal·scale sum and the breakdown is
    //!    expected to reconcile to the LP objective to within floating-point round-off.

    use std::path::Path;
    use std::sync::mpsc;

    use cobre_core::{TrainingEvent, scenario::ScenarioSource};
    use cobre_sddp::{
        SimulationScenarioResult, StudySetup, aggregate_simulation,
        hydro_models::prepare_hydro_models,
        setup::{StudyParams, prepare_stochastic},
    };
    use cobre_solver::highs::HighsSolver;

    use super::common::StubComm;

    fn case_dir(suffix: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic")
            .join(suffix)
    }

    fn train_and_simulate(suffix: &str) -> Vec<SimulationScenarioResult> {
        let dir = case_dir(suffix);
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

        let params = StudyParams::from_config(&config_with_sim)
            .expect("StudyParams::from_config must succeed");
        let construction = params.into_construction_config();

        let sentinel = Path::new("config.json");
        let training_source = config_with_sim
            .training_scenario_source(sentinel)
            .expect("training_scenario_source must parse");
        let simulation_source = config_with_sim
            .simulation_scenario_source(sentinel)
            .expect("simulation_scenario_source must parse");

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
        let mut solver = HighsSolver::new().expect("HighsSolver::new must succeed");
        let (event_tx, _event_rx) = mpsc::channel::<TrainingEvent>();

        let outcome = setup
            .train(
                &mut solver,
                &comm,
                1,
                HighsSolver::new,
                Some(event_tx),
                None,
            )
            .expect("train must return Ok");
        assert!(outcome.error.is_none(), "{suffix}: training error");
        let result = outcome.result;

        let mut pool = setup
            .create_workspace_pool(&comm, 1, HighsSolver::new)
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
        let scenario_results = drain_handle.join().expect("drain thread must not panic");

        let sim_config = setup.simulation_config();
        let _summary = aggregate_simulation(&local_costs.costs, sim_config, &comm)
            .expect("aggregate_simulation must succeed");

        scenario_results
    }

    /// The non-uniform block schedule shared by D33 and D34: the interior stages
    /// differ from stage 0's single block, which is the bug trigger.
    const BLOCK_COUNTS: [usize; 3] = [1, 3, 2];

    /// Assert one hydro record per (block, hydro) pair at every stage: a wrong
    /// per-stage stride does not change the count, but the count is the coarse shape
    /// guard the value reconciliation below complements.
    fn assert_per_block_equipment_shape(scenarios: &[SimulationScenarioResult], label: &str) {
        let scenario = scenarios.first().expect("one simulation scenario");
        for stage in &scenario.stages {
            let s = stage.stage_id as usize;
            let expected_blocks = BLOCK_COUNTS[s];
            // One hydro in both fixtures; per-block branch emits `n_blks` records.
            let block_ids: Vec<u32> = stage.hydros.iter().filter_map(|h| h.block_id).collect();
            assert_eq!(
                block_ids.len(),
                expected_blocks,
                "{label}: stage {s} must emit {expected_blocks} per-block hydro records",
            );
            // Cross-check the two spillage read paths against each other. The
            // per-record `spillage_cost` (a Part-1 per-block `grid.flat` read of the
            // spillage column) summed across blocks must equal the cost result's
            // `spillage_cost` (a Part-2 `range_sum` over the spillage range). D33/D34
            // declare no diversion, so the cost result's `spillage_cost` is the pure
            // spillage range_sum (diversion contributes 0). Pre-fix, at the
            // non-uniform interior stages the per-block base and the range base/length
            // disagree (one strides off stage-0's block width per block, the other
            // sums stage-0's range), so the two paths read different columns and the
            // cross-check diverges; post-fix both read the stage-correct columns and
            // agree to round-off.
            let per_record_spillage_cost: f64 = stage.hydros.iter().map(|h| h.spillage_cost).sum();
            let cost = stage.costs.first().expect("one cost record per stage");
            let scale = cost.spillage_cost.abs().max(1.0);
            let abs_err = (per_record_spillage_cost - cost.spillage_cost).abs();
            assert!(
                abs_err <= 1e-6 * scale,
                "{label}: stage {s} per-block spillage cost {per_record_spillage_cost} \
                 disagrees with the cost-result spillage_cost {} (abs err {abs_err}); \
                 the per-block column read and the range_sum addressed different \
                 columns at a non-uniform-block stage",
                cost.spillage_cost,
            );
            for h in &stage.hydros {
                assert!(
                    h.spillage_m3s.is_finite(),
                    "{label}: stage {s} spillage must be finite",
                );
            }
        }
    }

    /// Reconciliation invariant: the sum of every cost-breakdown category equals
    /// `immediate_cost` at every stage. The interior stages (1, 2) have block counts
    /// differing from stage 0's, so this fails pre-fix (stage-0 ranges misbook the
    /// cost) and passes post-fix (stage-correct ranges).
    fn assert_cost_reconciliation(scenarios: &[SimulationScenarioResult], label: &str) {
        let scenario = scenarios.first().expect("one simulation scenario");
        for stage in &scenario.stages {
            let s = stage.stage_id as usize;
            let cost = stage.costs.first().expect("one cost record per stage");

            // `hydro_violation_cost` is the total of the per-constraint violation
            // costs (outflow/turbined/generation/evaporation/withdrawal); summing it
            // here together with the standalone categories covers every priced column
            // exactly once. `generic_violation_cost` and `curtailment_cost` use a
            // non-range formula, but D33/D34 declare neither generic constraints nor
            // NCS, so both are zero and do not perturb the reconciliation.
            let breakdown = cost.thermal_cost
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

            // Relative tolerance scaled by the immediate cost magnitude: the
            // breakdown and `immediate_cost` are the same objective·primal·scale sum
            // grouped differently, so they agree to floating-point round-off.
            let scale = cost.immediate_cost.abs().max(1.0);
            let abs_err = (breakdown - cost.immediate_cost).abs();
            assert!(
                abs_err <= 1e-6 * scale,
                "{label}: stage {s} cost breakdown {breakdown} does not reconcile to \
                 immediate_cost {} (abs err {abs_err}); a stage-0 equipment range was \
                 summed at a non-uniform-block stage",
                cost.immediate_cost,
            );
        }
    }

    /// Pin the reported `anticipated_decision_mw` at the active interior delivery
    /// stage. The decision column's base is the per-stage `thermal.end`
    /// (`n_blks`-dependent); reading it off the global stage-0 base lands on a
    /// thermal-generation column at a non-uniform stage, so the reported decision MW
    /// equals one of the thermal's per-block `generation_mw` values instead of the
    /// distinct decision primal. Post-fix the decision is a single per-plant scalar
    /// (identical across blocks) distinct from every per-block generation; pre-fix it
    /// collapses onto a generation column. D34's anticipated thermal (`K = 1`)
    /// commits at stage 0 and re-commits at stage 1 (3 blocks ≠ stage 0's 1), so
    /// stage 1 is the bug-exposing active delivery stage.
    fn assert_anticipated_decision_mw(scenarios: &[SimulationScenarioResult], label: &str) {
        let scenario = scenarios.first().expect("one simulation scenario");
        // Stage 1 is the active interior stage with a block count differing from
        // stage 0's (3 vs 1); the anticipated K=1 thermal has a live decision there.
        let active_stage = 1u32;
        let stage = scenario
            .stages
            .iter()
            .find(|s| s.stage_id == active_stage)
            .expect("D34 has an interior stage 1");
        assert_eq!(
            BLOCK_COUNTS[active_stage as usize], 3,
            "stage 1 must carry 3 blocks (≠ stage 0's 1) to exercise the bug",
        );

        let antic: Vec<_> = stage.thermals.iter().filter(|t| t.is_anticipated).collect();
        assert!(
            !antic.is_empty(),
            "{label}: stage {active_stage} must report an anticipated thermal",
        );

        // The decision is a per-plant-per-stage scalar: identical across all block
        // records of the same anticipated thermal, present, finite, and positive.
        let decision = antic[0]
            .anticipated_decision_mw
            .expect("active anticipated decision must be Some at the delivery stage");
        assert!(
            decision.is_finite() && decision > 0.0,
            "{label}: anticipated_decision_mw must be a positive finite scalar, got {decision}",
        );
        for t in &antic {
            let d = t
                .anticipated_decision_mw
                .expect("anticipated decision present for every block record at the active stage");
            assert!(
                (d - decision).abs() <= 1e-9,
                "{label}: anticipated_decision_mw must be identical across blocks of the \
                 same thermal (per-plant scalar), got {d} vs {decision}",
            );
            // The decisive base-correctness check: a stage-0-based read lands on a
            // thermal-generation column at this non-uniform stage, so the reported
            // decision would equal one of the per-block generation values. The
            // stage-correct decision column is distinct from every per-block
            // generation of the same thermal.
            assert!(
                (d - t.generation_mw).abs() > 1e-6,
                "{label}: anticipated_decision_mw {d} coincides with this thermal's \
                 per-block generation_mw {} at stage {active_stage} — the decision was \
                 read off the global stage-0 base onto a generation column",
                t.generation_mw,
            );
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn d33_non_uniform_blocks_per_block_equipment_shape() {
        let scenarios = train_and_simulate("d33-per-stage-block-counts");
        assert_per_block_equipment_shape(&scenarios, "D33");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn d33_non_uniform_blocks_cost_reconciles() {
        let scenarios = train_and_simulate("d33-per-stage-block-counts");
        assert_cost_reconciliation(&scenarios, "D33");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn d34_anticipated_non_uniform_blocks_per_block_equipment_shape() {
        let scenarios = train_and_simulate("d34-anticipated-varying-blocks");
        assert_per_block_equipment_shape(&scenarios, "D34");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn d34_anticipated_non_uniform_blocks_cost_reconciles() {
        // D34 exercises the anticipated-decision column range too: its base is the
        // per-stage `thermal.end`, so the reconciliation here additionally pins the
        // anticipated-decision range repoint at the interior delivery stages.
        let scenarios = train_and_simulate("d34-anticipated-varying-blocks");
        assert_cost_reconciliation(&scenarios, "D34");
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn d34_anticipated_decision_mw_reads_stage_correct_column() {
        let scenarios = train_and_simulate("d34-anticipated-varying-blocks");
        assert_anticipated_decision_mw(&scenarios, "D34");
    }
}

mod indexer_slim_migration_rejection {
    //! Deletion-regression grep gate for the removed indexer types.
    //!
    //! The role-(b) geometry descriptor `StageIndexer` and its `EquipmentCounts`
    //! constructor input were deleted: the state-vector concern lives on
    //! `StateLayout`, the non-state study shape on `StudyDimensions`, and the
    //! per-stage equipment geometry on `StageLayout`/`StageGeometry`. This gate scans
    //! the production sources under `src/` and asserts those deleted types do not
    //! reappear — a regression guard that the deletion stays deleted.
    //!
    //! The forbidden tokens are assembled from character arrays so this gate file
    //! does not match itself (the project grep-gate convention).

    use std::path::{Path, PathBuf};

    fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("read_dir src") {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.is_dir() {
                collect_rs(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }

    /// Assemble a forbidden token from chars so the gate does not self-match.
    fn token(chars: &[char]) -> String {
        chars.iter().collect()
    }

    #[test]
    fn deleted_indexer_types_stay_deleted() {
        let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        collect_rs(&src, &mut files);
        assert!(!files.is_empty(), "must scan at least one source file");

        // `StageIndexer` — the deleted role-(b) geometry descriptor type.
        let stage_indexer = token(&['S', 't', 'a', 'g', 'e', 'I', 'n', 'd', 'e', 'x', 'e', 'r']);
        // `EquipmentCounts` — the deleted constructor input bag.
        let equipment_counts = token(&[
            'E', 'q', 'u', 'i', 'p', 'm', 'e', 'n', 't', 'C', 'o', 'u', 'n', 't', 's',
        ]);

        let mut offenders: Vec<String> = Vec::new();
        for path in &files {
            let body = std::fs::read_to_string(path).expect("read source file");
            for (lineno, line) in body.lines().enumerate() {
                if line.contains(&stage_indexer) {
                    offenders.push(format!(
                        "{}:{}: deleted type StageIndexer reappeared",
                        path.display(),
                        lineno + 1
                    ));
                }
                if line.contains(&equipment_counts) {
                    offenders.push(format!(
                        "{}:{}: deleted type EquipmentCounts reappeared",
                        path.display(),
                        lineno + 1
                    ));
                }
            }
        }

        assert!(
            offenders.is_empty(),
            "deletion regression — a deleted indexer type reappeared in production \
             sources (the role-(b) geometry now lives on StageLayout/StageGeometry, \
             the state vector on StateLayout, the study shape on StudyDimensions):\n{}",
            offenders.join("\n")
        );
    }
}

mod fixture_operational_start_date_order {
    //! Guards that every shipped deterministic-case entity fixture assigns
    //! `operational_start_date` so the `(operational_start_date, name)` build order
    //! reproduces the ascending-`id` order. A future fixture edit that breaks the
    //! date-monotonic-with-id property (e.g. a shared sentinel date that lets the
    //! `name` tiebreak reorder a collection) fails here instead of silently moving a
    //! parity hash.

    use std::fs;
    use std::path::{Path, PathBuf};

    use serde_json::Value;

    const ENTITY_FILES: &[&str] = &[
        "buses.json",
        "hydros.json",
        "thermals.json",
        "lines.json",
        "non_controllable_sources.json",
        "pumping_stations.json",
        "energy_contracts.json",
    ];

    fn deterministic_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/deterministic")
    }

    fn entity_array(doc: &Value) -> Option<&Vec<Value>> {
        doc.as_object()?
            .iter()
            .find(|(key, value)| key.as_str() != "$schema" && value.is_array())
            .and_then(|(_, value)| value.as_array())
    }

    fn check_file(path: &Path) {
        let text = fs::read_to_string(path).expect("read entity fixture");
        let doc: Value = serde_json::from_str(&text).expect("parse entity fixture");
        let Some(entities) = entity_array(&doc) else {
            return;
        };
        if entities.is_empty() {
            return;
        }

        let mut by_id: Vec<&Value> = entities.iter().collect();
        by_id.sort_by_key(|e| e["id"].as_i64().expect("entity id"));

        let mut by_build: Vec<&Value> = entities.iter().collect();
        by_build.sort_by(|a, b| {
            let da = a["operational_start_date"]
                .as_str()
                .expect("operational_start_date present");
            let db = b["operational_start_date"]
                .as_str()
                .expect("operational_start_date present");
            let na = a["name"].as_str().expect("entity name");
            let nb = b["name"].as_str().expect("entity name");
            (da, na).cmp(&(db, nb))
        });

        let ids_by_id: Vec<i64> = by_id.iter().map(|e| e["id"].as_i64().unwrap()).collect();
        let ids_by_build: Vec<i64> = by_build.iter().map(|e| e["id"].as_i64().unwrap()).collect();
        assert_eq!(
            ids_by_build,
            ids_by_id,
            "{}: (operational_start_date, name) order must equal ascending-id order",
            path.display()
        );
    }

    #[test]
    fn deterministic_fixtures_preserve_id_order_under_build_sort() {
        let root = deterministic_root();
        let mut checked = 0usize;
        for case in fs::read_dir(&root).expect("read deterministic root") {
            let system = case.expect("case dir entry").path().join("system");
            if !system.is_dir() {
                continue;
            }
            for fname in ENTITY_FILES {
                let path = system.join(fname);
                if path.exists() {
                    check_file(&path);
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 0,
            "no deterministic entity fixtures found under {}",
            root.display()
        );
    }
}

mod policy_entity_manifest {
    //! Integration coverage for the embedded per-slot entity manifest written into
    //! `policy/cuts/stage_NNN.bin` by the shared `write_checkpoint`.
    //!
    //! Trains a deterministic case to a policy checkpoint through the same
    //! `write_checkpoint` both front ends call, then reads the cut files back and
    //! asserts the manifest classification, identity, and per-stage length — including
    //! the reduced-stage (`inflow_lags: false`) d43 case, whose pool drops its
    //! inflow-lag slots.

    use std::path::Path;

    use cobre_core::scenario::ScenarioSource;
    use cobre_sddp::{
        StudySetup,
        hydro_models::prepare_hydro_models,
        orchestration::{CheckpointParams, write_checkpoint},
        setup::prepare_stochastic,
    };
    use cobre_solver::ActiveSolver;

    use super::common::StubComm;

    /// `EntityType::HydroInflowLag` discriminant from `schemas/policy.fbs`.
    const ENTITY_TYPE_HYDRO_INFLOW_LAG: u8 = 1;
    /// `EntityType::AnticipatedThermalState` discriminant from `schemas/policy.fbs`.
    const ENTITY_TYPE_ANTICIPATED_THERMAL_STATE: u8 = 2;

    fn case_dir(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/deterministic")
            .join(name)
    }

    /// Train a case to a policy checkpoint via the shared `write_checkpoint`, then read
    /// it back. Returns `(checkpoint, per-pool cut_state_layout n_state)`.
    fn train_and_read_checkpoint(name: &str) -> (cobre_io::PolicyCheckpoint, Vec<usize>) {
        let dir = case_dir(name);
        let config = cobre_io::parse_config(&dir.join("config.json")).expect("config must parse");
        let system = cobre_io::load_case(&dir).expect("load_case must succeed");

        let pr = prepare_stochastic(system, &dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic must succeed");
        let system = pr.system;
        let stochastic = pr.stochastic;

        let hydro_models =
            prepare_hydro_models(&system, &dir, false).expect("prepare_hydro_models must succeed");

        let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models)
            .expect("StudySetup must build");

        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

        let outcome = setup
            .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(outcome.error.is_none(), "expected no training error");
        let result = outcome.result;

        let _training_output = setup.build_training_output(&result, &[]);

        // Each pool is sized to its stage's cut-state dimension at construction
        // (`pool_state_dimensions[t] == cut_state_layouts[t].n_state()`), so the pool's
        // own `state_dimension` is the authoritative per-stage manifest length.
        let pool_n_state: Vec<usize> = setup.fcf.pools.iter().map(|p| p.state_dimension).collect();

        let tmp = tempfile::tempdir().expect("tempdir must succeed");
        let policy_dir = tmp.path().join("policy");
        write_checkpoint(
            &policy_dir,
            &setup,
            &system,
            &result,
            &CheckpointParams {
                max_iterations: 100,
                forward_passes: 1,
                seed: 42,
                export_states: false,
            },
        )
        .expect("write_checkpoint must succeed");

        let checkpoint = cobre_io::read_policy_checkpoint(&policy_dir)
            .expect("read_policy_checkpoint must succeed");
        (checkpoint, pool_n_state)
    }

    /// All-enabled study (d03, two-hydro cascade, storage state only): every stage's
    /// manifest length equals that pool's cut-state dimension, and the storage slots
    /// carry `entity_type == 0` with the hydro ids in `system.hydros()` order and
    /// `subindex == 0`.
    #[test]
    fn all_enabled_manifest_storage_slots_carry_hydro_identity() {
        let (checkpoint, pool_n_state) = train_and_read_checkpoint("d03-two-hydro-cascade");

        assert!(
            !checkpoint.stage_cuts.is_empty(),
            "checkpoint must contain stage cut files"
        );
        for stage in &checkpoint.stage_cuts {
            let t = stage.stage_id as usize;
            assert_eq!(
                stage.entity_manifest.len(),
                pool_n_state[t],
                "stage {t} manifest length must equal the pool cut-state dimension"
            );
            // d03 is storage-only state (no inflow lags, no anticipated): every slot is
            // a storage slot, in system.hydros() order, subindex 0.
            for (i, slot) in stage.entity_manifest.iter().enumerate() {
                assert_eq!(
                    slot.entity_type, 0,
                    "stage {t} slot {i} must be HydroStorage"
                );
                assert_eq!(slot.subindex, 0, "stage {t} slot {i} storage subindex 0");
            }
            // The two cascade hydros have ids 0 and 1, in `system.hydros()` order.
            assert_eq!(stage.entity_manifest[0].entity_id, 0);
            assert_eq!(stage.entity_manifest[1].entity_id, 1);
        }
    }

    /// d43 reduced stage: the pool sized by a stage with `inflow_lags: false` produces
    /// a manifest whose length equals that pool's reduced cut-state dimension and which
    /// carries NO `HydroInflowLag` (type 1) slot, while the full-state pools do carry
    /// lag slots. The reduced pool is located empirically by its `cut_state_layouts`
    /// dimension (the pool-sizing off-by-one means the reduced projection is not at the
    /// `inflow_lags: false` stage's own index).
    #[test]
    fn d43_reduced_stage_manifest_drops_inflow_lag_slots() {
        let (checkpoint, pool_n_state) = train_and_read_checkpoint("d43-storage-only-cut");

        // Per-stage manifest length always equals the pool's cut-state dimension.
        for stage in &checkpoint.stage_cuts {
            let t = stage.stage_id as usize;
            assert_eq!(
                stage.entity_manifest.len(),
                pool_n_state[t],
                "stage {t} manifest length must equal cut_state_layouts[{t}].n_state()"
            );
        }

        // d43 has one hydro; the full-state pools carry one storage + lag slots, the
        // reduced pool carries one storage slot only. So the reduced pool is the one
        // with the strictly smallest cut-state dimension, and exactly one pool is
        // reduced.
        let min_dim = *pool_n_state.iter().min().expect("at least one pool");
        let max_dim = *pool_n_state.iter().max().expect("at least one pool");
        assert!(
            min_dim < max_dim,
            "d43 must have a reduced pool (min {min_dim}) and a full pool (max {max_dim})"
        );

        let reduced_stage = checkpoint
            .stage_cuts
            .iter()
            .find(|s| s.entity_manifest.len() == min_dim)
            .expect("a reduced-dimension stage must exist");
        assert!(
            reduced_stage
                .entity_manifest
                .iter()
                .all(|s| s.entity_type != ENTITY_TYPE_HYDRO_INFLOW_LAG),
            "the reduced stage manifest must contain no HydroInflowLag slot"
        );

        // A full-state stage must carry at least one inflow-lag slot (d43 fits a PAR
        // model, so the full pools have lag dimensions).
        let full_stage = checkpoint
            .stage_cuts
            .iter()
            .find(|s| s.entity_manifest.len() == max_dim)
            .expect("a full-dimension stage must exist");
        assert!(
            full_stage
                .entity_manifest
                .iter()
                .any(|s| s.entity_type == ENTITY_TYPE_HYDRO_INFLOW_LAG),
            "a full-state stage manifest must contain an inflow-lag slot"
        );
    }

    /// Anticipated K=2 study (d37): the manifest carries `AnticipatedThermalState`
    /// (type 2) slots whose `entity_id` is the anticipated plant (id 1) and whose
    /// `subindex` ring slots cover `0..k_max` (`{0, 1}` for K=2).
    #[test]
    fn anticipated_k2_manifest_has_thermal_state_slots() {
        let (checkpoint, _pool_n_state) =
            train_and_read_checkpoint("d37-anticipated-commissioning");

        let stage0 = checkpoint
            .stage_cuts
            .iter()
            .find(|s| s.stage_id == 0)
            .expect("stage 0 cut file must exist");

        let anticipated_slots: Vec<_> = stage0
            .entity_manifest
            .iter()
            .filter(|s| s.entity_type == ENTITY_TYPE_ANTICIPATED_THERMAL_STATE)
            .collect();

        assert_eq!(
            anticipated_slots.len(),
            2,
            "K=2 yields two anticipated ring slots on the single anticipated plant"
        );
        for slot in &anticipated_slots {
            assert_eq!(slot.entity_id, 1, "anticipated slot must own plant id 1");
        }
        let ring: Vec<u32> = anticipated_slots.iter().map(|s| s.subindex).collect();
        assert_eq!(ring, vec![0, 1], "ring slots must cover 0..k_max");
    }
}
