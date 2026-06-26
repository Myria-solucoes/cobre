//! A filling cascade — two reservoirs in the Filling phase at the same stages —
//! needs no special handling: each carries its OWN per-stage soft floor and the
//! two couple ONLY through normal cascade releases.
//!
//! ## What the parity hash cannot see
//!
//! The declaration-order parity baseline hashes hydro storage / water values and
//! cuts; it does NOT hash the filling soft-penalty slacks. The per-stage `σ_fill`
//! (`filling_target_violation_hm3`) is invisible to the hash. A regression that
//! coupled the two filling floors through a shared inter-reservoir target term —
//! instead of letting each floor read its OWN `V_target` trajectory — could still
//! hash-match while silently producing a wrong (shared) shortfall. This test
//! exercises the per-floor independence and the release-only cascade coupling
//! directly through the full train+simulate pipeline.
//!
//! ## Fixture (`d40-filling-cascade`)
//!
//! Cascade `H_up (id 0, filling) → H_down (id 1, filling) → H_sink (id 2, real
//! outlet)` plus an off-cascade non-filling control `H_ctrl (id 3)`. BOTH filling
//! hydros carry `entry_stage_id = 4` and `filling { start_stage_id = 1,
//! filling_min_rate_m3s = 12 }`, so over the 6-stage horizon both are PreFilling
//! at id 0, Filling at ids 1–3, and Operating at ids 4–5 — the one topology with
//! two reservoirs in the Filling phase simultaneously. Block counts change across
//! the 1/1/3/2/3/1 schedule, exercising the per-stage geometry and per-stage `τ`.
//! All hydros use `constant_productivity`.
//!
//! The volume-target model gives each Filling stage a per-stage soft floor
//! `v_out[t] + σ_fill[t] ≥ V_target[t]`, with the target anchored backward from
//! the dead volume: `V_target[3] = min_storage = 60`, `V_target[2] = 60 − ζ·rate =
//! 28.896` hm³, and `V_target[1] = 28.896 − ζ·rate = −2.208` clipped negative
//! (trivially satisfied ⇒ `σ_fill[1] = 0`). With `std_m3s = 0` everywhere the
//! trajectory is deterministic. The own incremental inflows are DISTINCT (`H_up`
//! 5 m³/s, `H_down` 3 m³/s over Filling) so each hydro's inflow-only accumulation
//! falls short of its `V_target` at the binding stages 2 and 3 by a DIFFERENT
//! amount — the per-floor independence is observable as two distinct `σ_fill`
//! values. During Filling both impound (turbine pinned `[0,0]`), so `H_up`'s
//! release onto `H_down` is ~0 at the Filling stages: the cascade couples
//! release-only, with the gap between `H_down`'s Δstorage and its incremental-only
//! balance equal to `H_up`'s routed release (an exact identity, no phantom
//! inter-floor term). The cascade outlet `H_sink` carries a small turbine
//! (`max_turbined 20 m³/s`) so the finite, zero-discount terminal horizon cannot
//! monetize a last-Filling-stage water dump as cheap hydro generation; without it
//! the LP would optimally spill the dead-volume water at id 3 (driving `σ_fill[3]`
//! to the full dead volume for BOTH and collapsing the independence) instead of
//! holding it. Inflows recover to 60 m³/s in Operating so both climb above the dead
//! volume and `σ^{v-} → 0`.

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

// ---------------------------------------------------------------------------
// Fixture constants (study stage ids == indices for this single-resolution horizon)
// ---------------------------------------------------------------------------

/// PreFilling → Filling boundary shared by both filling hydros.
const START_STAGE_ID: i32 = 1;
/// Filling → Operating boundary shared by both filling hydros.
const ENTRY_STAGE_ID: i32 = 4;
/// Hydro entity ids in canonical order. `H_up` (id 0) is the upstream filling
/// reservoir, `H_down` (id 1) the downstream filling reservoir fed by `H_up`,
/// `H_sink` (id 2) the real cascade outlet, and `H_ctrl` (id 3) the off-cascade
/// control.
const H_UP_ID: i32 = 0;
const H_DOWN_ID: i32 = 1;
const H_SINK_ID: i32 = 2;
const H_CTRL_ID: i32 = 3;

/// Shared dead volume (filling target / soft operating floor), hm³. Mirrors
/// `system/hydros.json` `min_storage_hm3` for both filling hydros.
const MIN_STORAGE_HM3: f64 = 60.0;
/// Shared per-stage minimum accumulation rate (m³/s). Mirrors `system/hydros.json`
/// `filling.filling_min_rate_m3s` for both filling hydros.
const FILLING_MIN_RATE_M3S: f64 = 12.0;
/// `H_up`'s own incremental inflow over the Filling stages (m³/s). Mirrors
/// `scenarios/inflow_seasonal_stats.parquet`, id 0 at stages 1–3.
const H_UP_FILLING_INCR_M3S: f64 = 5.0;
/// `H_down`'s own incremental inflow over the Filling stages (m³/s). Distinct from
/// `H_up`'s so the two `σ_fill` shortfalls differ. Mirrors id 1 at stages 1–3.
const H_DOWN_FILLING_INCR_M3S: f64 = 3.0;
/// `H_ctrl`'s reservoir bounds (hm³), mirroring `system/hydros.json`.
const H_CTRL_MIN_STORAGE_HM3: f64 = 0.0;
const H_CTRL_MAX_STORAGE_HM3: f64 = 200.0;
/// `H_ctrl`'s stage-0 seed (hm³), mirroring `initial_conditions.json`.
const H_CTRL_SEED_HM3: f64 = 100.0;

/// m³/s → hm³ per hour. Mirrors `crate::lp_builder::M3S_TO_HM3` (a private const
/// not exported to integration tests). `ζ = total_stage_hours · M3S_TO_HM3`.
const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;
/// Every stage in `stages.json` totals 720 h, so `ζ` is uniform.
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

/// Per-(stage, hydro) view collapsing the stage's per-block rows to the scalars
/// the assertions need. Storage and the σ_fill / σ^{v-} slacks are stage-level
/// (block-invariant), so they are read from the first block; turbined flow is
/// summed across blocks. The cascade water balance reads the per-block rows
/// directly (it needs per-block τ), not this aggregate.
struct StageHydro {
    storage_initial_hm3: f64,
    storage_final_hm3: f64,
    incremental_inflow_m3s: f64,
    generation_mw: f64,
    turbined_total_m3s: f64,
    filling_target_violation_hm3: f64,
    storage_violation_below_hm3: f64,
}

/// Collect the per-stage hydro view for `hydro_id` from a scenario result. Storage
/// and slack fields are stage-level (block-invariant); turbined flow is summed
/// across blocks (the cascade balance needs the whole-stage release).
fn stage_hydro(
    scenario: &cobre_sddp::SimulationScenarioResult,
    hydro_id: i32,
    stage_index: usize,
) -> StageHydro {
    let rows: Vec<_> = scenario.stages[stage_index]
        .hydros
        .iter()
        .filter(|r| r.hydro_id == hydro_id)
        .collect();
    assert!(
        !rows.is_empty(),
        "hydro {hydro_id} must have at least one row at stage {stage_index}"
    );
    let first = rows[0];
    StageHydro {
        storage_initial_hm3: first.storage_initial_hm3,
        storage_final_hm3: first.storage_final_hm3,
        incremental_inflow_m3s: first.incremental_inflow_m3s,
        generation_mw: rows.iter().map(|r| r.generation_mw).sum(),
        turbined_total_m3s: rows.iter().map(|r| r.turbined_m3s).sum(),
        filling_target_violation_hm3: first.filling_target_violation_hm3,
        storage_violation_below_hm3: first.storage_violation_below_hm3,
    }
}

/// `H_up`'s whole-stage release (turbine + spillage) routed onto `H_down`'s
/// water-balance row at `stage_index`, in hm³. The release term uses per-block
/// `τ_k = block_hours·M3S_TO_HM3` built from the per-block rows so the balance is
/// exact regardless of the stage's block count (it differs across the 1/1/3/2/3/1
/// schedule). Using a single stage-level ζ on the release would mis-weight a
/// multi-block stage.
fn routed_release_hm3(
    scenario: &cobre_sddp::SimulationScenarioResult,
    upstream_id: i32,
    stage_index: usize,
) -> f64 {
    scenario.stages[stage_index]
        .hydros
        .iter()
        .filter(|r| r.hydro_id == upstream_id)
        .map(|r| {
            (r.turbined_m3s + r.spillage_m3s) * block_hours(stage_index, r.block_id) * M3S_TO_HM3
        })
        .sum()
}

/// Train the d40 filling-cascade case, simulate one deterministic scenario, and
/// assert the per-floor independence and release-only cascade coupling the parity
/// hash cannot see.
// Rationale: the assertions read a single train+simulate run; splitting the body
// into per-criterion helpers would force the one expensive pipeline to be re-run
// per helper (or thread the whole scenario + outcome through opaque arguments),
// obscuring the linear "train → simulate → assert each criterion against the same
// trajectory" flow this regression exists to make legible.
#[allow(clippy::too_many_lines)]
#[test]
fn filling_cascade_floors_are_independent_and_couple_release_only() {
    let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d40-filling-cascade");

    // Load + parse succeed and BOTH filling configs round-trip with the shared
    // window — the overlapping Filling phase is the case's whole point.
    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");
    let system_for_check = cobre_io::load_case(&case_dir).expect("load_case must succeed");
    for (id, downstream) in [(H_UP_ID, H_DOWN_ID), (H_DOWN_ID, H_SINK_ID)] {
        let h = system_for_check
            .hydros()
            .iter()
            .find(|h| h.id.0 == id)
            .unwrap_or_else(|| panic!("filling hydro {id} must be present"));
        let filling = h
            .filling
            .as_ref()
            .unwrap_or_else(|| panic!("hydro {id} filling must be Some"));
        assert_eq!(
            filling.start_stage_id, START_STAGE_ID,
            "hydro {id} filling start_stage_id"
        );
        assert_eq!(
            h.entry_stage_id,
            Some(ENTRY_STAGE_ID),
            "hydro {id} entry_stage_id"
        );
        assert_eq!(
            h.downstream_id.map(|d| d.0),
            Some(downstream),
            "hydro {id} cascade edge"
        );
    }

    // The shipped case disables simulation (the parity harness trains only).
    // Enable one deterministic simulation scenario so the extraction path runs.
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

    let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();
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
        "training error (an infeasible filling stage is a real data finding, not \
         a bound to relax): {:?}",
        outcome.error
    );

    let lower_bounds: Vec<f64> = event_rx
        .try_iter()
        .filter_map(|e| {
            if let TrainingEvent::ConvergenceUpdate { lower_bound, .. } = e {
                Some(lower_bound)
            } else {
                None
            }
        })
        .collect();
    assert!(
        !lower_bounds.is_empty(),
        "must capture at least one ConvergenceUpdate lower bound"
    );
    for window in lower_bounds.windows(2) {
        let (prev, next) = (window[0], window[1]);
        let tol = 1e-6 * prev.abs().max(1.0);
        assert!(
            next >= prev - tol,
            "lower bound must be monotone within FP tolerance: {prev} -> {next} \
             (allowed slack {tol})"
        );
    }

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
        .expect("simulate must not return Err (the soft slacks keep every stage feasible)");

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
        6,
        "one stage record per study stage (horizon ids 0..=5)",
    );

    let zeta = STAGE_HOURS * M3S_TO_HM3;
    // Shared V_target trajectory (anchored backward from the dead volume). Both
    // filling hydros share min_storage and rate, so the targets coincide; the
    // shortfalls differ only because realized storage differs (distinct inflows).
    let v_target_3 = MIN_STORAGE_HM3;
    let v_target_2 = MIN_STORAGE_HM3 - zeta * FILLING_MIN_RATE_M3S;

    // ── Per-floor INDEPENDENCE ────────────────────────────────────────────────
    // Both H_up and H_down have a strictly-positive σ_fill at the binding Filling
    // stages 2 and 3, and each σ_fill equals its OWN independently-computed
    // shortfall against its OWN realized storage and the shared V_target. The two
    // hydros' shortfalls DIFFER (distinct own incrementals), proving the floors
    // read each hydro's own trajectory rather than a single shared floor. A
    // terminal-only check would miss a per-stage floor regression that binds only
    // at the last Filling stage; a single-hydro check would miss a shared-floor
    // bug that fuses the two reservoirs' shortfalls.
    for (filling_stage, v_target) in [(2_usize, v_target_2), (3, v_target_3)] {
        let h_up_s = stage_hydro(scenario, H_UP_ID, filling_stage);
        let h_down_s = stage_hydro(scenario, H_DOWN_ID, filling_stage);
        let sigma_up = h_up_s.filling_target_violation_hm3;
        let sigma_down = h_down_s.filling_target_violation_hm3;

        // Each σ_fill is strictly positive: the short inflow leaves storage below
        // V_target.
        assert!(
            sigma_up > 1e-6,
            "H_up σ_fill must be strictly positive at Filling stage {filling_stage} \
             (storage {} below V_target {v_target}); got {sigma_up}",
            h_up_s.storage_final_hm3
        );
        assert!(
            sigma_down > 1e-6,
            "H_down σ_fill must be strictly positive at Filling stage {filling_stage} \
             (storage {} below V_target {v_target}); got {sigma_down}",
            h_down_s.storage_final_hm3
        );

        // Each σ_fill equals its OWN shortfall V_target − v_out against its own
        // realized storage: σ_fill = max(0, V_target − v_out). This is the
        // per-floor contract — the floor row reads this hydro's storage, not the
        // sibling's.
        let shortfall_up = (v_target - h_up_s.storage_final_hm3).max(0.0);
        let shortfall_down = (v_target - h_down_s.storage_final_hm3).max(0.0);
        assert!(
            (sigma_up - shortfall_up).abs() < 1e-3,
            "H_up σ_fill must equal its OWN shortfall at stage {filling_stage}: \
             σ_fill {sigma_up} vs V_target − v_out = {shortfall_up} (storage {})",
            h_up_s.storage_final_hm3
        );
        assert!(
            (sigma_down - shortfall_down).abs() < 1e-3,
            "H_down σ_fill must equal its OWN shortfall at stage {filling_stage}: \
             σ_fill {sigma_down} vs V_target − v_out = {shortfall_down} (storage {})",
            h_down_s.storage_final_hm3
        );

        // The two shortfalls DIFFER: a shared floor would force identical σ_fill.
        assert!(
            (sigma_up - sigma_down).abs() > 1e-3,
            "H_up and H_down σ_fill must DIFFER at stage {filling_stage} (distinct own \
             incrementals {H_UP_FILLING_INCR_M3S} vs {H_DOWN_FILLING_INCR_M3S} m³/s leave \
             distinct storages and so distinct shortfalls); a shared floor would force them \
             equal: σ_up {sigma_up} vs σ_down {sigma_down}"
        );
    }

    // σ^{v-} → 0 for BOTH at the last Operating stage (id 5): inflow recovers and
    // storage climbs above the dead volume, so the soft operating floor releases.
    for id in [H_UP_ID, H_DOWN_ID] {
        let sigma_floor_s5 = stage_hydro(scenario, id, 5).storage_violation_below_hm3;
        assert!(
            sigma_floor_s5 < 1e-6,
            "hydro {id} σ^{{v-}} must be ~0 at the last Operating stage (id 5) after the \
             inflow recovers and storage climbs above the dead volume {MIN_STORAGE_HM3}; got \
             {sigma_floor_s5}"
        );
    }

    // ── Release-only cascade coupling ─────────────────────────────────────────
    // At each Filling stage the closed water balance on H_down is
    //     Δstorage = ζ·incr_down − release_down + ROUTED,
    // where ROUTED = release_up routed from the upstream filling reservoir (the
    // only coupling). Rearranged, the GAP between H_down's actual Δstorage and its
    // incremental-only balance (ζ·incr_down − release_down) IS exactly H_up's
    // routed release. Asserting gap == release_up (within tolerance) proves the
    // reservoirs couple ONLY through the cascade release: a phantom inter-floor
    // routing term would make the gap exceed release_up. The criterion floor is
    // `gap ≥ release_up − tol`; the tight equality is the stronger statement.
    for filling_stage in [1_usize, 2, 3] {
        let h_down_s = stage_hydro(scenario, H_DOWN_ID, filling_stage);
        let delta_storage = h_down_s.storage_final_hm3 - h_down_s.storage_initial_hm3;
        let release_down_hm3: f64 = scenario.stages[filling_stage]
            .hydros
            .iter()
            .filter(|r| r.hydro_id == H_DOWN_ID)
            .map(|r| {
                (r.turbined_m3s + r.spillage_m3s)
                    * block_hours(filling_stage, r.block_id)
                    * M3S_TO_HM3
            })
            .sum();
        let incremental_balance = zeta * h_down_s.incremental_inflow_m3s - release_down_hm3;
        let routed_gap = delta_storage - incremental_balance;
        let release_up = routed_release_hm3(scenario, H_UP_ID, filling_stage);
        assert!(
            routed_gap >= release_up - 1e-3,
            "Filling stage {filling_stage}: H_down's routed-water gap ({routed_gap:.6} hm³ = \
             Δstorage {delta_storage:.6} − incremental-only balance {incremental_balance:.6}) \
             must be at least H_up's routed release ({release_up:.6} hm³)"
        );
        // Tight equality: no coupling beyond the cascade release (release-only).
        assert!(
            (routed_gap - release_up).abs() < 1e-3,
            "Filling stage {filling_stage}: H_down's routed-water gap ({routed_gap:.6} hm³) must \
             EQUAL H_up's routed release ({release_up:.6} hm³) — a larger gap means a phantom \
             inter-floor coupling term landed on H_down's balance row"
        );
    }

    // ── Continuous handoff for BOTH filling hydros ────────────────────────────
    for id in [H_UP_ID, H_DOWN_ID] {
        let s3_final = stage_hydro(scenario, id, 3).storage_final_hm3;
        let s4_initial = stage_hydro(scenario, id, 4).storage_initial_hm3;
        assert!(
            (s3_final - s4_initial).abs() < 1e-6,
            "hydro {id}: end-of-Filling storage at id 3 ({s3_final}) must equal incoming storage \
             at id 4 ({s4_initial}) via the pin chain"
        );
    }

    // ── Off-cascade control dispatches as a normal Operating plant ────────────
    let mut h_ctrl_moved = false;
    for stage in 0..6 {
        let h_ctrl_s = stage_hydro(scenario, H_CTRL_ID, stage);
        for v in [h_ctrl_s.storage_initial_hm3, h_ctrl_s.storage_final_hm3] {
            assert!(
                (H_CTRL_MIN_STORAGE_HM3 - 1e-6..=H_CTRL_MAX_STORAGE_HM3 + 1e-6).contains(&v),
                "control H_ctrl storage must stay within [{H_CTRL_MIN_STORAGE_HM3}, \
                 {H_CTRL_MAX_STORAGE_HM3}] at stage {stage}; got {v}"
            );
        }
        if (h_ctrl_s.storage_final_hm3 - H_CTRL_SEED_HM3).abs() > 1e-6 {
            h_ctrl_moved = true;
        }
    }
    assert!(
        h_ctrl_moved,
        "control H_ctrl must dispatch (its storage moves off the {H_CTRL_SEED_HM3} hm³ seed), not \
         stay frozen like a PreFilling hydro"
    );

    // Before entry both filling hydros neither generate nor turbine: the
    // turbine/generation columns are pinned [0, 0] until Operating.
    for id in [H_UP_ID, H_DOWN_ID] {
        for stage in 0..(ENTRY_STAGE_ID as usize) {
            let s = stage_hydro(scenario, id, stage);
            assert!(
                s.generation_mw.abs() < 1e-6,
                "hydro {id} must not generate before entry (stage {stage}); got {}",
                s.generation_mw
            );
            assert!(
                s.turbined_total_m3s.abs() < 1e-6,
                "hydro {id} must not turbine before entry (stage {stage}); got {}",
                s.turbined_total_m3s
            );
        }
    }
}

/// Block duration in hours for `(stage_index, block_id)`, mirroring `stages.json`.
/// Used by the per-block-exact cascade release balance. `block_id` is the
/// `SimulationHydroResult.block_id` (always `Some` for a hydro with turbines).
fn block_hours(stage_index: usize, block_id: Option<u32>) -> f64 {
    let blk = block_id.expect("turbine-branch hydro rows carry a block id") as usize;
    match stage_index {
        0 | 1 | 5 => 720.0,
        2 | 4 => [240.0, 240.0, 240.0][blk],
        3 => [360.0, 360.0][blk],
        _ => unreachable!("d40 has stages 0..=5"),
    }
}
