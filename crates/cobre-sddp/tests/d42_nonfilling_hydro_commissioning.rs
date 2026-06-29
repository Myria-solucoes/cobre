//! A non-filling hydro that commissions mid-horizon reuses the `PreFilling`
//! short-circuit reformulation: while commissioning-dormant its dam is not built,
//! so its turbine/spillage/generation columns are pinned `[0, 0]`, its storage is
//! frozen at the inert initial-condition value, and its incremental inflow flows
//! past the un-built site — onto the first active downstream reservoir, or to the
//! sink at a cascade tail. The danger this case guards against is the trapped-water
//! trap: zeroing the flow columns while leaving the inflow on the hydro's OWN
//! balance row makes the LP infeasible whenever the site has inflow.
//!
//! ## Fixture (`d42-nonfilling-hydro-commissioning`)
//!
//! Three NON-filling hydros (ids 0, 1, 2) over a 4-stage horizon, single FLAT
//! block, 720 h each, deterministic (std = 0):
//!
//! - `H_new` (id 0): `entry_stage_id = 2`, downstream = `H_down` (id 1). Dormant at
//!   stages 0-1 (the trapped-water catcher: nonzero 40 m³/s local inflow routed onto
//!   a downstream reservoir), Operating at 2-3.
//! - `H_down` (id 1): no commissioning window, real reservoir outlet
//!   (`downstream_id = null`). Operating every stage; receives `H_new`'s routed
//!   inflow while `H_new` is dormant.
//! - `H_tail` (id 2): `entry_stage_id = 2`, cascade tail (`downstream_id = null`).
//!   Dormant at 0-1 (the sink-discard fixture: its inflow exits the system), then
//!   Operating at 2-3.
//!
//! A 300 MW thermal backstop keeps every stage feasible while the hydros are
//! dormant, so an infeasibility here is a real trapped-water regression, not a
//! tuning artifact.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

use std::path::Path;
use std::sync::mpsc;

use cobre_core::TrainingEvent;
use cobre_core::scenario::ScenarioSource;
use cobre_sddp::{StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic};
use cobre_solver::ActiveSolver;

mod common;
use common::StubComm;

/// Hydro entity ids in canonical order. Mirrors `system/hydros.json`.
const H_NEW_ID: i32 = 0;
const H_DOWN_ID: i32 = 1;
const H_TAIL_ID: i32 = 2;

/// First commissioned stage for `H_new` and `H_tail` (`entry_stage_id`). Mirrors
/// `system/hydros.json`.
const ENTRY_STAGE_ID: usize = 2;

/// `H_new`'s and `H_tail`'s own incremental inflow (m³/s), constant across stages.
/// Mirrors `scenarios/inflow_seasonal_stats.parquet` ids 0 and 2.
const H_NEW_INCR_M3S: f64 = 40.0;
/// `H_down`'s own incremental inflow (m³/s). Mirrors id 1.
const H_DOWN_INCR_M3S: f64 = 20.0;
/// Initial-condition storage seeds (hm³). Mirrors `initial_conditions.json`; the
/// dormant hydros pin to their IC value (NOT `[0, 0]`, NOT `min_storage`).
const H_NEW_SEED_HM3: f64 = 50.0;
const H_TAIL_SEED_HM3: f64 = 30.0;

/// m³/s → hm³ per hour. Mirrors `crate::lp_builder::M3S_TO_HM3` (private). `ζ =
/// total_stage_hours · M3S_TO_HM3`; every stage totals 720 h.
const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
const STAGE_HOURS: f64 = 720.0;

/// Per-(stage, hydro) view: single FLAT block per stage, so each scalar reads from
/// the lone block row.
struct StageHydro {
    turbined_m3s: f64,
    spillage_m3s: f64,
    generation_mw: f64,
    incremental_inflow_m3s: f64,
    storage_initial_hm3: f64,
    storage_final_hm3: f64,
}

fn hydro_rows(
    scenario: &cobre_sddp::SimulationScenarioResult,
    hydro_id: i32,
    stage_index: usize,
) -> Vec<&cobre_sddp::SimulationHydroResult> {
    scenario.stages[stage_index]
        .hydros
        .iter()
        .filter(|r| r.hydro_id == hydro_id)
        .collect()
}

fn stage_hydro(
    scenario: &cobre_sddp::SimulationScenarioResult,
    hydro_id: i32,
    stage_index: usize,
) -> StageHydro {
    let rows = hydro_rows(scenario, hydro_id, stage_index);
    assert!(
        !rows.is_empty(),
        "hydro {hydro_id} must have at least one row at stage {stage_index}"
    );
    let first = rows[0];
    StageHydro {
        turbined_m3s: rows.iter().map(|r| r.turbined_m3s).sum(),
        spillage_m3s: rows.iter().map(|r| r.spillage_m3s).sum(),
        generation_mw: rows.iter().map(|r| r.generation_mw).sum(),
        incremental_inflow_m3s: first.incremental_inflow_m3s,
        storage_initial_hm3: first.storage_initial_hm3,
        storage_final_hm3: first.storage_final_hm3,
    }
}

/// Train the d42 case, simulate one deterministic scenario, and assert the
/// dormant-non-filling reformulation: feasibility (no trapped water), the routed
/// inflow landing on the downstream reservoir, the sink discard, and the
/// dormant-vs-active output transition at `entry`.
// Rationale: the four assertion blocks share one train+simulate run; splitting them
// would re-run the pipeline per block or thread the scenario through opaque args.
#[allow(clippy::too_many_lines)]
#[test]
fn nonfilling_commissioning_routes_inflow_and_is_feasible() {
    let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d42-nonfilling-hydro-commissioning");

    // Load + parse succeed: a non-filling hydro carrying a commissioning window is
    // valid (the relaxed filling⟹entry guard), and the three windows round-trip.
    let system_for_check = cobre_io::load_case(&case_dir).expect("load_case must succeed");
    for (id, entry, downstream) in [
        (H_NEW_ID, Some(ENTRY_STAGE_ID as i32), Some(H_DOWN_ID)),
        (H_DOWN_ID, None, None),
        (H_TAIL_ID, Some(ENTRY_STAGE_ID as i32), None),
    ] {
        let h = system_for_check
            .hydros()
            .iter()
            .find(|h| h.id.0 == id)
            .unwrap_or_else(|| panic!("hydro {id} must be present"));
        assert!(h.filling.is_none(), "hydro {id} must be non-filling");
        assert_eq!(h.entry_stage_id, entry, "hydro {id} entry_stage_id");
        assert_eq!(
            h.downstream_id.map(|d| d.0),
            downstream,
            "hydro {id} cascade edge"
        );
    }

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
    assert!(
        outcome.error.is_none(),
        "training must be feasible — an infeasibility here means the dormant hydro's \
         inflow was trapped on its own balance row instead of short-circuited: {:?}",
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
        .expect(
            "simulate must not return Err (the dormant reformulation keeps every stage feasible)",
        );

    drop(result_tx);
    let scenario_results = drain_handle.join().expect("drain thread must not panic");
    assert_eq!(scenario_results.len(), 1, "one deterministic scenario");
    let scenario = &scenario_results[0];
    assert_eq!(
        scenario.stages.len(),
        4,
        "one record per study stage (0..=3)"
    );

    let zeta = STAGE_HOURS * M3S_TO_HM3;

    // ── Dormant outputs before entry (the [0,0] flow columns + frozen IC storage) ─
    // H_new and H_tail neither turbine nor generate while dormant, and their storage
    // stays pinned at the inert IC seed (frozen identity v = v_in = seed). A nonzero
    // turbine/generation here would mean the window was not applied; a storage drift
    // would mean the storage column was not decoupled.
    for stage in 0..ENTRY_STAGE_ID {
        for (id, seed) in [(H_NEW_ID, H_NEW_SEED_HM3), (H_TAIL_ID, H_TAIL_SEED_HM3)] {
            // The simulation read path emits a dormant output row (not an absent
            // entity): exactly one row per single-FLAT-block stage, with the
            // non-line `operative_state_code == 1`, mirroring the existing
            // commissioning_active simulation gate for thermals/lines/NCS.
            let rows = hydro_rows(scenario, id, stage);
            assert_eq!(
                rows.len(),
                1,
                "dormant hydro {id} must emit exactly one output row at stage {stage} (present, not absent)"
            );
            assert_eq!(
                rows[0].operative_state_code, 1,
                "dormant hydro {id} output row at stage {stage} carries operative_state_code 1"
            );
            let s = stage_hydro(scenario, id, stage);
            assert!(
                s.turbined_m3s.abs() < 1e-6,
                "dormant hydro {id} must not turbine at stage {stage}; got {}",
                s.turbined_m3s
            );
            assert!(
                s.spillage_m3s.abs() < 1e-6,
                "dormant hydro {id} must not spill at stage {stage}; got {}",
                s.spillage_m3s
            );
            assert!(
                s.generation_mw.abs() < 1e-6,
                "dormant hydro {id} must not generate at stage {stage}; got {}",
                s.generation_mw
            );
            assert!(
                (s.storage_final_hm3 - seed).abs() < 1e-6,
                "dormant hydro {id} storage must stay frozen at the IC seed {seed} at stage \
                 {stage} (decoupled, NOT [0,0] / min_storage); got {}",
                s.storage_final_hm3
            );
        }
    }

    // ── Trapped-water catcher: H_new's inflow lands on H_down's balance row ───────
    // While H_new (id 0) is dormant, the river flows past its site onto the active
    // downstream reservoir H_down (id 1). The closed water balance on H_down is
    //     Δstorage = ζ·incr_down − release_down + ROUTED,
    // and ROUTED is exactly ζ·incr_new (H_new's incremental inflow, since H_new's own
    // release is 0 while dormant). Rearranged, the GAP between H_down's actual
    // Δstorage and its incremental-only balance IS ζ·incr_new. A trapped-water bug
    // (inflow stranded on H_new's own row) would make the gap 0 instead.
    for stage in 0..ENTRY_STAGE_ID {
        let h_down = stage_hydro(scenario, H_DOWN_ID, stage);
        let release_down_hm3 = (h_down.turbined_m3s + h_down.spillage_m3s) * zeta;
        let delta_storage = h_down.storage_final_hm3 - h_down.storage_initial_hm3;
        let incremental_balance = zeta * h_down.incremental_inflow_m3s - release_down_hm3;
        let routed_gap = delta_storage - incremental_balance;
        let expected_routed = zeta * H_NEW_INCR_M3S;
        assert!(
            (routed_gap - expected_routed).abs() < 1e-3,
            "stage {stage}: H_down's routed-water gap ({routed_gap:.6} hm³ = Δstorage \
             {delta_storage:.6} − incremental-only balance {incremental_balance:.6}) must EQUAL \
             ζ·incr_new ({expected_routed:.6} hm³); a smaller gap means H_new's inflow was \
             trapped on its own frozen row instead of routed downstream"
        );
        // Sanity: H_down actually carries its own inflow (the routing is additive, not
        // a relabeling).
        assert_eq!(
            h_down.incremental_inflow_m3s, H_DOWN_INCR_M3S,
            "stage {stage}: H_down keeps its own incremental inflow"
        );
    }

    // ── Sink discard: H_tail's dormant inflow exits the system ────────────────────
    // H_tail (id 2) has no downstream while dormant, so its incremental inflow is
    // discarded at the sink and its storage stays frozen — already asserted above as
    // part of the dormant-output block. The feasibility of the whole solve (asserted
    // via outcome.error.is_none()) is the sink case's "no infeasibility" guarantee:
    // a trapped sink inflow would have made stage 0/1 infeasible.
    for stage in 0..ENTRY_STAGE_ID {
        let h_tail = stage_hydro(scenario, H_TAIL_ID, stage);
        assert_eq!(
            h_tail.incremental_inflow_m3s, H_NEW_INCR_M3S,
            "stage {stage}: H_tail carries its own (discarded) incremental inflow"
        );
    }

    // ── Active from entry: no Filling phase, normal operation ─────────────────────
    // From entry onward H_new and H_tail are Operating: they can turbine/generate
    // (the columns are no longer pinned [0,0]) and their storage is no longer frozen
    // at the seed. At least one commissioned stage must show the plant dispatching to
    // prove the window opened straight to Operating, and storage must be able to move
    // off the dormant seed (the frozen identity is released at entry).
    let mut new_dispatched = false;
    let mut tail_dispatched = false;
    let mut new_storage_moved = false;
    for stage in ENTRY_STAGE_ID..4 {
        let s_new = stage_hydro(scenario, H_NEW_ID, stage);
        let s_tail = stage_hydro(scenario, H_TAIL_ID, stage);
        if s_new.turbined_m3s > 1e-6 || s_new.generation_mw > 1e-6 {
            new_dispatched = true;
        }
        if s_tail.turbined_m3s > 1e-6 || s_tail.generation_mw > 1e-6 {
            tail_dispatched = true;
        }
        if (s_new.storage_final_hm3 - H_NEW_SEED_HM3).abs() > 1e-6 {
            new_storage_moved = true;
        }
    }
    assert!(
        new_dispatched,
        "H_new must dispatch at some commissioned stage (>= entry {ENTRY_STAGE_ID}) — the window \
         opens straight to Operating with no Filling phase"
    );
    assert!(
        tail_dispatched,
        "H_tail must dispatch at some commissioned stage (>= entry {ENTRY_STAGE_ID})"
    );
    assert!(
        new_storage_moved,
        "H_new's storage must move off the dormant IC seed {H_NEW_SEED_HM3} hm³ once Operating — \
         the frozen-identity storage decoupling is released at entry, not held like a PreFilling pin"
    );
}
