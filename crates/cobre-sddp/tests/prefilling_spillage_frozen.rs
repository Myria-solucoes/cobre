//! A `PreFilling` hydro upstream of an active reservoir must have spillage exactly
//! `0` and contribute zero phantom water to the downstream balance row.
//!
//! ## The bug the one-sided floor cannot catch
//!
//! `fill_spillage_columns` once left every hydro's spillage free `[0, +∞)`. A
//! `PreFilling` hydro's spillage column is decoupled from its frozen storage (its
//! real inflow already left via the short-circuit), so a free column lets the
//! optimizer inject water the hydro does not have onto the first active downstream
//! hydro's water-balance row — a conservation violation. The downstream hydro
//! turbines that phantom water cheaply and displaces thermal generation.
//!
//! The `d38` simulation regression checks H3's routed inflow with a ONE-SIDED floor
//! (`routed_gap >= ζ·incr_H2`): phantom spillage from H2 only inflates the gap, so
//! the floor still passes and the bug hides. This test pins the structural fix
//! directly — H2's spillage is `0` while `PreFilling` — and pins the downstream gap
//! with a TWO-SIDED equality that no phantom water can satisfy.
//!
//! ## Fixture (`d38-dead-volume-filling`)
//!
//! Cascade `H1 (id 0) → H2 (id 1, filling) → H3 (id 2, real fed downstream)`. H2's
//! `filling.start_stage_id = 2` makes it `PreFilling` at stage ids 0–1 (the dam is
//! not built yet), with 15 m³/s of its own incremental inflow at each. Stages 0 and
//! 1 are single FLAT blocks of 720 h. H2 is upstream of the active reservoir H3, so
//! these two stages are exactly the "PreFilling hydro upstream of an active
//! reservoir" geometry the "spillage frozen [0,0] during PreFilling" contract targets.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::doc_markdown
)]

use std::path::Path;
use std::sync::mpsc;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::TrainingEvent;
use cobre_core::scenario::ScenarioSource;
use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
use cobre_solver::ActiveSolver;

/// Hydro ids in canonical order. H2 (id 1) is the filling hydro that is PreFilling
/// at stages 0–1; H1 (id 0) is its upstream feeder; H3 (id 2) is the active fed
/// downstream that the short-circuit routes onto.
const H1_ID: i32 = 0;
const H2_ID: i32 = 1;
const H3_ID: i32 = 2;

/// Last PreFilling stage index for H2 (exclusive upper bound of the loop):
/// `filling.start_stage_id = 2` ⇒ PreFilling at ids 0 and 1.
const H2_START_STAGE_ID: usize = 2;

/// H2's own incremental inflow (m³/s) at PreFilling stages 0–1. Mirrors
/// `scenarios/inflow_seasonal_stats.parquet`, hydro 1 ids 0–1.
const H2_PREFILLING_INCR_M3S: f64 = 15.0;

/// H3's own incremental inflow (m³/s), constant across stages. Mirrors
/// `scenarios/inflow_seasonal_stats.parquet`, hydro 2.
const H3_INCR_M3S: f64 = 20.0;

/// m³/s → hm³ per hour. Mirrors `crate::lp_builder::M3S_TO_HM3` (private). Stages
/// 0 and 1 are single FLAT blocks totalling 720 h, so `ζ = 720 · M3S_TO_HM3`.
const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
const STAGE_HOURS: f64 = 720.0;

/// Single-rank communicator stub that faithfully copies data through the
/// collectives, so the pipeline runs without MPI.
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

/// Sum a hydro's release (turbine + spillage, m³/s) across the blocks of one stage.
fn release_m3s(
    scenario: &cobre_sddp::SimulationScenarioResult,
    hydro_id: i32,
    stage_index: usize,
) -> (f64, f64) {
    let turbined = scenario.stages[stage_index]
        .hydros
        .iter()
        .filter(|r| r.hydro_id == hydro_id)
        .map(|r| r.turbined_m3s)
        .sum();
    let spillage = scenario.stages[stage_index]
        .hydros
        .iter()
        .filter(|r| r.hydro_id == hydro_id)
        .map(|r| r.spillage_m3s)
        .sum();
    (turbined, spillage)
}

/// First-block scalar (storage is block-invariant) for a hydro at one stage.
fn storage_initial_final(
    scenario: &cobre_sddp::SimulationScenarioResult,
    hydro_id: i32,
    stage_index: usize,
) -> (f64, f64, f64) {
    let first = scenario.stages[stage_index]
        .hydros
        .iter()
        .find(|r| r.hydro_id == hydro_id)
        .unwrap_or_else(|| panic!("hydro {hydro_id} must have a row at stage {stage_index}"));
    (
        first.storage_initial_hm3,
        first.storage_final_hm3,
        first.incremental_inflow_m3s,
    )
}

/// Train + simulate the d38 fixture and assert the PreFilling spillage freeze: H2
/// (PreFilling at ids 0–1, upstream of the active H3) spills exactly `0`, and H3's
/// routed-water gap equals ζ·incr_H2 PLUS H1's actual re-routed release — a
/// two-sided equality no phantom spillage can satisfy.
#[test]
fn prefilling_hydro_upstream_of_active_reservoir_has_zero_spillage_and_no_phantom_water() {
    let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d38-dead-volume-filling");

    let mut config =
        cobre_io::parse_config(&case_dir.join("config.json")).expect("config must parse");
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

    let hydro_models =
        prepare_hydro_models(&system, &case_dir, false).expect("prepare_hydro_models must succeed");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup::new");

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new");

    let (event_tx, _event_rx) = mpsc::channel::<TrainingEvent>();
    let outcome = setup
        .train(
            &mut solver,
            &comm,
            1,
            ActiveSolver::new,
            Some(event_tx),
            None,
        )
        .expect("train must not return Err");
    assert!(outcome.error.is_none(), "training must be feasible");

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
    assert_eq!(scenario_results.len(), 1, "one deterministic scenario");
    let scenario = &scenario_results[0];

    let zeta = STAGE_HOURS * M3S_TO_HM3;

    for stage in 0..H2_START_STAGE_ID {
        // (1) The PreFilling hydro spills exactly 0. This is the assertion the d38
        // one-sided routed-water floor cannot make — it never inspects H2's own
        // spillage, only the aggregate gap on H3.
        let (h2_turb, h2_spill) = release_m3s(scenario, H2_ID, stage);
        assert!(
            h2_spill.abs() < 1e-6,
            "PreFilling H2 must not spill at stage {stage} (no dam exists to spill from); got \
             {h2_spill} m³/s — a positive value is phantom water injected downstream"
        );
        assert!(
            h2_turb.abs() < 1e-6,
            "PreFilling H2 must not turbine at stage {stage}; got {h2_turb} m³/s"
        );

        // (2) No phantom water downstream: H3's closed water balance over the stage
        // is Δstorage = ζ·incr_H3 − release_H3 + ROUTED, where the only routed terms
        // onto H3's row while H2 is PreFilling are H2's own incremental inflow
        // (ζ·incr_H2) and H1's re-routed release (H1→H3, since H1→H2→H3 collapses
        // while H2 is absent). Stages 0–1 are single FLAT blocks, so ζ weights every
        // release uniformly. The gap must EQUAL that sum — a TWO-SIDED equality.
        // Phantom spillage from H2 would add ζ·spill_H2 to the gap and break it; the
        // d38 floor (`gap >= ζ·incr_H2`) would still pass.
        let (h3_init, h3_final, h3_incr) = storage_initial_final(scenario, H3_ID, stage);
        let (h3_turb, h3_spill) = release_m3s(scenario, H3_ID, stage);
        let (h1_turb, h1_spill) = release_m3s(scenario, H1_ID, stage);

        let release_h3 = (h3_turb + h3_spill) * zeta;
        let routed_h1 = (h1_turb + h1_spill) * zeta;
        let delta_storage = h3_final - h3_init;
        let incremental_balance = zeta * h3_incr - release_h3;
        let routed_gap = delta_storage - incremental_balance;
        let expected_routed = zeta * H2_PREFILLING_INCR_M3S + routed_h1;

        assert!(
            (routed_gap - expected_routed).abs() < 1e-3,
            "stage {stage}: H3's routed-water gap ({routed_gap:.6} hm³ = Δstorage \
             {delta_storage:.6} − incremental-only balance {incremental_balance:.6}) must EQUAL \
             ζ·incr_H2 + H1's re-routed release ({expected_routed:.6} hm³); a LARGER gap means H2 \
             injected phantom spillage onto H3's row (the bug the one-sided floor hides)"
        );

        // Sanity: H3 keeps its own incremental inflow (routing is additive).
        assert!(
            (h3_incr - H3_INCR_M3S).abs() < 1e-6,
            "stage {stage}: H3 incremental inflow must be {H3_INCR_M3S} m³/s (own inflow must \
             not be stripped by PreFilling routing); got {h3_incr}"
        );
    }
}
