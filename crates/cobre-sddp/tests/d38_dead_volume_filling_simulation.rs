//! Hydro dead-volume filling must reach the simulation output and bind under
//! short inflow.
//!
//! ## What the parity hash cannot see
//!
//! The declaration-order parity baseline hashes hydro storage / water values and
//! cuts; it does NOT hash the filling soft-penalty slacks. The per-stage
//! `σ_fill` (`filling_target_violation_hm3`) and the operating soft-floor
//! `σ^{v-}` (`storage_violation_below_hm3`) are invisible to the hash, as is the
//! PreFilling cascade short-circuit that routes an absent dam's water onto its
//! real downstream's water-balance row. A gating bug that let the slacks stay
//! zero under a short fill, or dropped the routed water into a sink, would still
//! hash-match. This test exercises those paths directly through the full
//! train+simulate pipeline.
//!
//! ## Fixture (`d38-dead-volume-filling`)
//!
//! Cascade `H1 (id 0) → H2 (id 1, filling) → H3 (id 2, real fed downstream)` plus
//! an off-cascade non-filling control `H4 (id 3)`. `H2` carries
//! `entry_stage_id = 4` and `filling { start_stage_id = 2, filling_min_rate_m3s = 12 }`,
//! so over the 6-stage horizon it is PreFilling at ids 0–1, Filling at ids 2–3,
//! and Operating at ids 4–5. Block counts change at BOTH phase boundaries
//! (schedule 1/1/3/2/3/1: 1→3 at stage 1→2, 2→3 at stage 3→4), exercising the
//! per-stage geometry and per-stage `τ`. All hydros use `constant_productivity`
//! (FPHA-during-filling exclusion is out of scope here; it is unit-tested
//! separately). A single filling hydro is the design baseline — the chained
//! transitive short-circuit walk is covered by `filling_cut_validity`.
//!
//! The volume-target filling model gives each Filling stage a per-stage SOFT
//! floor `v_out[t] + σ_fill[t] ≥ V_target[t]` (`fill_filling_target_rows`:
//! row_upper = +∞), with the target anchored backward from the dead volume:
//! `V_target[3] = min_storage_hm3 = 60` and `V_target[2] = 60 − ζ·rate = 28.896`
//! hm³. The end-of-Filling storage at id 3 hands off to the incoming storage at id
//! 4 with no reset (the pin chain). Because the floor is soft and Filling spillage
//! is FREE, H2 is not forced to hold its water across Filling: with the downstream
//! H3 starved (the PreFilling phantom-spill removed), the LP optimally releases the
//! impounded water at the last Filling stage (id 3) to feed H3, driving `σ_fill[3]`
//! to the full dead-volume shortfall. d40 (byte-identical penalties) suppresses the
//! same release only via a downstream turbine cap that d38 lacks. The inflow
//! schedule (`std_m3s = 0` everywhere ⇒ deterministic) keeps `σ_fill[t] > 0` at
//! BOTH filling stages (ids 2 and 3); inflows RECOVER in Operating (60 m³/s) so H2
//! climbs above the dead volume and `σ^{v-} → 0` by id 5. `H1` is starved before
//! Operating (seed 0, inflow 2/2/0/0 m³/s over ids 0–3).

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
use cobre_sddp::{
    SolverStatsDelta, StudySetup, hydro_models::prepare_hydro_models, setup::prepare_stochastic,
};
use cobre_solver::ActiveSolver;

// ---------------------------------------------------------------------------
// Fixture constants (study stage ids == indices for this single-resolution horizon)
// ---------------------------------------------------------------------------

/// PreFilling → Filling boundary for `H2`: filling begins at this stage id.
const START_STAGE_ID: i32 = 2;
/// Filling → Operating boundary for `H2`: it becomes a normal plant at this id.
const ENTRY_STAGE_ID: i32 = 4;
/// Hydro entity ids in canonical order. `H1` (id 0) is the upstream cascade
/// feeder, `H2` (id 1) the mid-cascade filling hydro, `H3` (id 2) its real fed
/// downstream, and `H4` (id 3) the off-cascade control.
const H1_ID: i32 = 0;
const H2_ID: i32 = 1;
const H3_ID: i32 = 2;
const H4_ID: i32 = 3;

/// `H2`'s dead volume (filling target / soft operating floor), hm³. Mirrors
/// `system/hydros.json` `H2.reservoir.min_storage_hm3`.
const H2_MIN_STORAGE_HM3: f64 = 60.0;
/// `H2`'s per-stage minimum accumulation rate (m³/s). Mirrors
/// `system/hydros.json` `H2.filling.filling_min_rate_m3s`.
const H2_FILLING_MIN_RATE_M3S: f64 = 12.0;
/// `H4`'s reservoir bounds (hm³), mirroring `system/hydros.json`.
const H4_MIN_STORAGE_HM3: f64 = 0.0;
const H4_MAX_STORAGE_HM3: f64 = 200.0;
/// `H4`'s stage-0 seed (hm³), mirroring `initial_conditions.json`.
const H4_SEED_HM3: f64 = 100.0;

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
/// the assertions need. The simulation emits one `SimulationHydroResult` per
/// (block, hydro): storage and the σ_fill / σ^{v-} slacks are stage-level
/// (block-invariant), so they are read from the first block; generation and
/// turbined flow are summed across blocks. The §7.1 water balance reads the
/// per-block rows directly (it needs per-block τ), not this aggregate.
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
/// and slack fields are stage-level (block-invariant); turbined/spillage are
/// summed across blocks (the §7.1 balance needs the whole-stage release).
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

/// Train the d38 dead-volume-filling case, simulate one deterministic scenario,
/// and assert the §7.1–§7.7 obligations the parity hash cannot see.
// Rationale: the seven §7 assertions read a single train+simulate run; splitting
// the body into per-criterion helpers would force the one expensive pipeline to be
// re-run per helper (or thread the whole scenario + outcome through opaque
// arguments), obscuring the linear "train → simulate → assert each criterion against
// the same trajectory" flow this regression exists to make legible.
#[allow(clippy::too_many_lines)]
#[test]
fn dead_volume_filling_binds_and_routes_in_simulation() {
    let case_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/deterministic/d38-dead-volume-filling");

    // §7.1: load + parse succeed and the filling config round-trips.
    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");
    let system_for_check = cobre_io::load_case(&case_dir).expect("load_case must succeed");
    let h2 = system_for_check
        .hydros()
        .iter()
        .find(|h| h.id.0 == H2_ID)
        .expect("H2 must be present");
    let filling = h2.filling.as_ref().expect("H2.filling must be Some");
    assert_eq!(
        filling.start_stage_id, START_STAGE_ID,
        "H2 filling start_stage_id"
    );
    assert_eq!(h2.entry_stage_id, Some(ENTRY_STAGE_ID), "H2 entry_stage_id");
    assert_eq!(
        h2.downstream_id.map(|d| d.0),
        Some(H3_ID),
        "H2 must short-circuit onto H3 (its real fed downstream)"
    );

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

    // §7.6: monotone lower bound within absolute-relative FP tolerance.
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

    // §7.6: zero basis-rejection spike at the two filling boundary stages. Stage
    // id == index here, so the boundary stages are START_STAGE_ID and
    // ENTRY_STAGE_ID. Aggregate ONLY those two stages on the forward/backward
    // passes — a global aggregate would dilute a boundary spike.
    let boundary_stages = [START_STAGE_ID, ENTRY_STAGE_ID];
    let boundary_rejections = SolverStatsDelta::aggregate(
        outcome
            .result
            .solver_stats_log
            .iter()
            .filter(|entry| {
                matches!(entry.phase, "forward" | "backward")
                    && boundary_stages.contains(&entry.stage)
            })
            .map(|entry| &entry.delta),
    )
    .basis_consistency_failures;
    assert_eq!(
        boundary_rejections, 0,
        "filling boundaries: expected 0 basis rejections at PreFilling->Filling \
         (id {START_STAGE_ID}) and Filling->Operating (id {ENTRY_STAGE_ID}); got \
         {boundary_rejections} (a rejection means a cut row/column was relocated \
         across the phase change, breaking reconstruct_basis slot-identity matching)"
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

    // §7.2: zero generation and turbine for H2 at every stage before entry
    // (PreFilling 0–1 and Filling 2–3). The turbine/generation columns are pinned
    // [0, 0] until Operating.
    for stage in 0..(ENTRY_STAGE_ID as usize) {
        let h2_s = stage_hydro(scenario, H2_ID, stage);
        assert!(
            h2_s.generation_mw.abs() < 1e-6,
            "H2 must not generate before entry (stage {stage}); got {}",
            h2_s.generation_mw
        );
        assert!(
            h2_s.turbined_total_m3s.abs() < 1e-6,
            "H2 must not turbine before entry (stage {stage}); got {}",
            h2_s.turbined_total_m3s
        );
    }

    // §7.3: H2's PreFilling freeze and its per-stage Filling water balance. H2
    // spillage is frozen [0,0] only while PreFilling (no dam yet — the "spillage
    // frozen during PreFilling" contract); it is FREE during Filling, so the LP may
    // legitimately release the impounded water at the last Filling stage (id 3) to
    // feed the water-starved downstream H3 rather than hold it. The σ_fill row is a
    // SOFT floor (`fill_filling_target_rows`: row_upper = +∞), so storage is NOT
    // required to be monotone across Filling. d40 (byte-identical penalties)
    // suppresses the same release only via a downstream turbine cap that d38 lacks.

    // (a) PreFilling freeze: storage stays pinned at the empty-pit seed through both
    // PreFilling stages (ids 0–1), the stage-0 incoming storage.
    let h2_s0 = stage_hydro(scenario, H2_ID, 0);
    let h2_s1 = stage_hydro(scenario, H2_ID, 1);
    let seed = h2_s0.storage_initial_hm3;
    assert!(
        seed.abs() < 1e-6,
        "H2 PreFilling seed must be the empty pit 0.0; got {seed}"
    );
    for (stage, h2) in [(0_usize, &h2_s0), (1, &h2_s1)] {
        assert!(
            (h2.storage_final_hm3 - seed).abs() < 1e-6,
            "H2 storage must be frozen at the seed through PreFilling stage {stage}; got {}",
            h2.storage_final_hm3
        );
    }

    // (b) Per-stage Filling water balance (ids 2–3), every term pinned to FP
    // tolerance: v_out = v_in + ζ·incr_H2 − τ·spillage_H2 + routed_upstream(H1→H2).
    // H2 turbine and diversion are 0 while Filling (asserted by §7.2 / the
    // diversion-pin contract), so only its spillage subtracts. The upstream H1
    // release enters with per-block τ_k = block_hours·M3S_TO_HM3, summed across the
    // stage's blocks so the balance is exact regardless of block count.
    let h2_incr_filling_m3s = 5.0; // scenarios/inflow_seasonal_stats.parquet, H2 ids 2–3
    for stage in (START_STAGE_ID as usize)..(ENTRY_STAGE_ID as usize) {
        let h2 = stage_hydro(scenario, H2_ID, stage);
        let h2_spill_release_hm3: f64 = scenario.stages[stage]
            .hydros
            .iter()
            .filter(|r| r.hydro_id == H2_ID)
            .map(|r| r.spillage_m3s * block_hours(stage, r.block_id) * M3S_TO_HM3)
            .sum();
        let routed_upstream_hm3: f64 = scenario.stages[stage]
            .hydros
            .iter()
            .filter(|r| r.hydro_id == H1_ID)
            .map(|r| {
                (r.turbined_m3s + r.spillage_m3s) * block_hours(stage, r.block_id) * M3S_TO_HM3
            })
            .sum();
        let expected_v_out = h2.storage_initial_hm3 + zeta * h2_incr_filling_m3s
            - h2_spill_release_hm3
            + routed_upstream_hm3;
        assert!(
            (h2.storage_final_hm3 - expected_v_out).abs() < 1e-3,
            "Filling stage {stage}: H2's closed water balance must hold — v_out {} = v_in {} + \
             ζ·incr {} − spill_release {h2_spill_release_hm3} + routed_upstream {routed_upstream_hm3}; \
             expected {expected_v_out}",
            h2.storage_final_hm3,
            h2.storage_initial_hm3,
            zeta * h2_incr_filling_m3s
        );
    }

    // (c) When H2 releases at the last Filling stage (v_out[3] < v_out[2] with
    // spillage[3] > 0), the released water must reach the starved downstream H3 (the
    // same routed-gap balance §7.1 uses on H3), and σ_fill[3] must book the
    // dead-volume shortfall exactly: σ_fill[3] = V_target[3] − v_out[3].
    let h2_s2 = stage_hydro(scenario, H2_ID, 2);
    let h2_s3 = stage_hydro(scenario, H2_ID, 3);
    let h2_s3_spill_total: f64 = scenario.stages[3]
        .hydros
        .iter()
        .filter(|r| r.hydro_id == H2_ID)
        .map(|r| r.spillage_m3s)
        .sum();
    if h2_s3.storage_final_hm3 < h2_s2.storage_final_hm3 - 1e-6 {
        assert!(
            h2_s3_spill_total > 1e-6,
            "H2 storage fell across Filling ids 2→3 ({} → {}); the only free release path while \
             Filling is spillage, so spillage[3] must be > 0; got {h2_s3_spill_total}",
            h2_s2.storage_final_hm3,
            h2_s3.storage_final_hm3
        );
        let h3_s3 = stage_hydro(scenario, H3_ID, 3);
        let h3_release_hm3: f64 = scenario.stages[3]
            .hydros
            .iter()
            .filter(|r| r.hydro_id == H3_ID)
            .map(|r| (r.turbined_m3s + r.spillage_m3s) * block_hours(3, r.block_id) * M3S_TO_HM3)
            .sum();
        let h3_delta = h3_s3.storage_final_hm3 - h3_s3.storage_initial_hm3;
        let h3_incremental_balance = zeta * h3_s3.incremental_inflow_m3s - h3_release_hm3;
        let h3_routed_gap = h3_delta - h3_incremental_balance;
        let h2_released_hm3: f64 = scenario.stages[3]
            .hydros
            .iter()
            .filter(|r| r.hydro_id == H2_ID)
            .map(|r| r.spillage_m3s * block_hours(3, r.block_id) * M3S_TO_HM3)
            .sum();
        assert!(
            h3_routed_gap >= h2_released_hm3 - 1e-3,
            "H2's last-Filling release ({h2_released_hm3:.6} hm³ of spillage) must reach H3: H3's \
             routed-water gap ({h3_routed_gap:.6} hm³) must be at least H2's released volume — \
             conservation, not a vanished dump"
        );
        let v_target_3 = H2_MIN_STORAGE_HM3;
        assert!(
            (h2_s3.filling_target_violation_hm3 - (v_target_3 - h2_s3.storage_final_hm3)).abs()
                < 1e-3,
            "σ_fill[3] must book the dead-volume shortfall exactly: σ_fill {} = V_target[3] {} − \
             v_out[3] {}",
            h2_s3.filling_target_violation_hm3,
            v_target_3,
            h2_s3.storage_final_hm3
        );
    }

    // §7.1 (water balance on H3, re-anchored — does NOT use inflow_m3s): during
    // PreFilling the dam at H2 does not exist, so H2's water short-circuits onto
    // H3's real water-balance row. The closed water balance on H3 is
    //     Δstorage = ζ·incr_H3 − release_H3 + ROUTED,
    // where ROUTED = ζ·incr_H2 + (any H1 release re-routed H1→H3) + H2's withdrawal.
    // H1 is starved before Operating, so its PreFilling release is small or zero;
    // the asserted floor therefore rests on H2's OWN incremental alone. Rearranged,
    // the GAP between H3's actual Δstorage and its incremental-only balance IS the
    // routed contribution: it must be strictly positive, proving the water lands on
    // H3's row (H3 is a fed reservoir, not a sink). It must moreover be at least
    // H2's own routed incremental ζ·incr_H2 — a known, deterministic quantity
    // (std=0) — which is a quantitative floor on the routing, not just
    // `> incremental_only`.
    let h2_prefilling_incr_m3s = 15.0; // mirrors scenarios/inflow_seasonal_stats.parquet, H2 ids 0–1
    let routed_floor_hm3 = zeta * h2_prefilling_incr_m3s;
    for prefilling_stage in [0_usize, 1] {
        let h3_s = stage_hydro(scenario, H3_ID, prefilling_stage);
        let delta_storage = h3_s.storage_final_hm3 - h3_s.storage_initial_hm3;
        // The release term uses per-block τ_k = block_hours·M3S_TO_HM3, built from
        // the per-block rows so the balance is exact regardless of the stage's
        // block count (it differs across the 1/1/3/2/3/1 schedule). Using a single
        // stage-level ζ on the release would mis-weight a multi-block stage.
        let release_hm3: f64 = scenario.stages[prefilling_stage]
            .hydros
            .iter()
            .filter(|r| r.hydro_id == H3_ID)
            .map(|r| {
                (r.turbined_m3s + r.spillage_m3s)
                    * block_hours(prefilling_stage, r.block_id)
                    * M3S_TO_HM3
            })
            .sum();
        let incremental_balance = zeta * h3_s.incremental_inflow_m3s - release_hm3;
        let routed_gap = delta_storage - incremental_balance;
        assert!(
            routed_gap >= routed_floor_hm3 - 1e-3,
            "§7.1 PreFilling stage {prefilling_stage}: the routed-water gap on H3 \
             ({routed_gap:.6} hm³ = Δstorage {delta_storage:.6} − incremental-only balance \
             {incremental_balance:.6}) must be at least H2's own routed incremental \
             ζ·incr_H2 = {routed_floor_hm3:.6} hm³; a smaller gap means short-circuited water \
             was dropped before reaching H3's row (a sink bug). H3 incremental_inflow = {} m³/s, \
             release = {release_hm3:.6} hm³",
            h3_s.incremental_inflow_m3s
        );
    }

    // §7.4: σ_fill binds at BOTH Filling stages (ids 2 and 3) under the short
    // inflow, and σ^{v-} → 0 by the last Operating stage (id 5) on recovery. The
    // volume-target model imposes a per-stage soft floor v_out[t] + σ_fill[t] ≥
    // V_target[t]; with the short inflow the inflow-only accumulation (12.96 hm³
    // by id 2, 25.92 hm³ by id 3) stays below the backward-anchored targets
    // (V_target[2] = min_storage − ζ·rate = 28.896, V_target[3] = min_storage =
    // 60.0), so σ_fill is strictly positive at each filling stage. A terminal-only
    // check would miss a per-stage floor regression that binds only at the last
    // Filling stage.
    let v_target_2 = H2_MIN_STORAGE_HM3 - zeta * H2_FILLING_MIN_RATE_M3S;
    for (filling_stage, v_target) in [(2_usize, v_target_2), (3, H2_MIN_STORAGE_HM3)] {
        let sigma_fill = stage_hydro(scenario, H2_ID, filling_stage).filling_target_violation_hm3;
        assert!(
            sigma_fill > 1e-6,
            "§7.4: H2 σ_fill must be strictly positive at Filling stage {filling_stage} (the \
             short inflow leaves storage below V_target[{filling_stage}] = {v_target}); got \
             {sigma_fill}"
        );
    }
    let sigma_floor_s5 = stage_hydro(scenario, H2_ID, 5).storage_violation_below_hm3;
    assert!(
        sigma_floor_s5 < 1e-6,
        "§7.4: H2 σ^{{v-}} must be ~0 at the last Operating stage (id 5) after the inflow \
         recovers and storage climbs above the dead volume {H2_MIN_STORAGE_HM3}; got \
         {sigma_floor_s5}"
    );

    // §7.5: entry handoff / pin chain — H2's end-of-Filling storage at id 3 equals
    // its incoming storage at id 4 (first block).
    let h2_s3_final = stage_hydro(scenario, H2_ID, 3).storage_final_hm3;
    let h2_s4_initial = stage_hydro(scenario, H2_ID, 4).storage_initial_hm3;
    assert!(
        (h2_s3_final - h2_s4_initial).abs() < 1e-6,
        "§7.5: end-of-Filling storage at id 3 ({h2_s3_final}) must equal incoming storage at \
         id 4 ({h2_s4_initial}) via the pin chain"
    );

    // §7.7: off-cascade control H4 dispatches as a normal Operating plant — every
    // storage value within [min, max] and at least one value off the seed.
    let mut h4_moved = false;
    for stage in 0..6 {
        let h4_s = stage_hydro(scenario, H4_ID, stage);
        for v in [h4_s.storage_initial_hm3, h4_s.storage_final_hm3] {
            assert!(
                (H4_MIN_STORAGE_HM3 - 1e-6..=H4_MAX_STORAGE_HM3 + 1e-6).contains(&v),
                "control H4 storage must stay within [{H4_MIN_STORAGE_HM3}, \
                 {H4_MAX_STORAGE_HM3}] at stage {stage}; got {v}"
            );
        }
        if (h4_s.storage_final_hm3 - H4_SEED_HM3).abs() > 1e-6 {
            h4_moved = true;
        }
    }
    assert!(
        h4_moved,
        "control H4 must dispatch (its storage moves off the {H4_SEED_HM3} hm³ seed), not \
         stay frozen like a PreFilling hydro"
    );
}

/// Block duration in hours for `(stage_index, block_id)`, mirroring `stages.json`.
/// Used by the §7.1 per-block-exact release balance. `block_id` is the
/// `SimulationHydroResult.block_id` (always `Some` for a hydro with turbines).
fn block_hours(stage_index: usize, block_id: Option<u32>) -> f64 {
    let blk = block_id.expect("turbine-branch hydro rows carry a block id") as usize;
    match stage_index {
        0 | 1 | 5 => 720.0,
        2 | 4 => [240.0, 240.0, 240.0][blk],
        3 => [360.0, 360.0][blk],
        _ => unreachable!("d38 has stages 0..=5"),
    }
}
