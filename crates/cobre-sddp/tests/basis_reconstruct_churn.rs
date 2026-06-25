//! Cross-churn regression tests for [`cobre_sddp::basis_reconstruct`]: guards
//! slot reconciliation, `state_at_capture` routing, and the iteration-wide
//! hot-path allocation invariants against LML1 deactivation, budget eviction,
//! and new-cut churn.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::too_many_lines
)]

use std::path::Path;
use std::sync::mpsc;

use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
use cobre_core::scenario::ScenarioSource;
use cobre_io::config::{SelectionMethod, StoppingRuleConfig};
use cobre_sddp::{
    SolverStatsDelta, SolverStatsLogEntry, StudySetup, hydro_models::prepare_hydro_models,
    setup::prepare_stochastic,
};
use cobre_solver::ActiveSolver;

// ---------------------------------------------------------------------------
// StubComm — single-rank communicator for testing
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

fn d03_case_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/<crate> must have a parent")
        .parent()
        .expect("crates/ must have a parent (repo root)")
        .join("examples/deterministic/d03-two-hydro-cascade")
}

fn d01_case_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/<crate> must have a parent")
        .parent()
        .expect("crates/ must have a parent (repo root)")
        .join("examples/deterministic/d01-thermal-dispatch")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Aggregate `SolverStatsDelta` across all `"forward"` log entries.
fn sum_forward_deltas(log: &[SolverStatsLogEntry]) -> SolverStatsDelta {
    SolverStatsDelta::aggregate(
        log.iter()
            .filter(|e| e.phase == "forward")
            .map(|e| &e.delta),
    )
}

// ---------------------------------------------------------------------------
// Test 1: Cross-churn — LML1 + budget + new cuts
// ---------------------------------------------------------------------------

/// All three churn types active at once; the simplex-iteration pin catches a
/// `padding_state = x_hat` regression (it raises the count beyond the ±5 % band).
#[test]
fn basis_reconstruct_churn() {
    // Simplex pin and lower-bound pin are calibrated against HiGHS's simplex path
    // and last-ULP accumulation, so they gate to the `highs` backend; the
    // structural checks below run on both. Regenerate the pin only if HiGHS major
    // version, the fixture parameters, or the G1 design changes.
    #[cfg(feature = "highs")]
    const PINNED_SIMPLEX_ITERS: u64 = 30;
    #[cfg(feature = "highs")]
    const LO_SIMPLEX: u64 = PINNED_SIMPLEX_ITERS * 95 / 100;
    #[cfg(feature = "highs")]
    const HI_SIMPLEX: u64 = PINNED_SIMPLEX_ITERS * 105 / 100;

    #[cfg(feature = "highs")]
    const PINNED_FINAL_LB: f64 = 1_391_697.766_666_666_8;

    let case_dir = d03_case_dir();
    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");

    config.training.forward_passes = Some(3);
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit { limit: 8 }]);

    config.training.cut_selection.selection = Some(SelectionMethod::Lml1 {
        tie_tolerance: 1e-10,
        check_frequency: 1,
    });

    // Budget = 6 = 2 iters × 3 fwd-passes, so eviction first fires after iteration 2.
    config.training.cut_selection.max_active_per_stage = Some(6);

    let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
    let prepare_result =
        prepare_stochastic(system, &case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, &case_dir, false).expect("prepare_hydro_models must succeed");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "basis_reconstruct_churn: expected no training error, got: {:?}",
        outcome.error
    );
    let result = outcome.result;
    assert_eq!(
        result.iterations, 8,
        "basis_reconstruct_churn: expected 8 iterations, got {}",
        result.iterations
    );

    let fwd = sum_forward_deltas(&result.solver_stats_log);

    // basis_reconstructions is excluded from forward-log MPI packing, so it is
    // always 0 in these entries; basis_consistency_failures == 0 is the proxy
    // for "reconstruction is active and producing valid bases".
    assert_eq!(
        fwd.basis_consistency_failures, 0,
        "basis_reconstruct_churn: expected 0 basis rejections, got {} \
         (reconstructed bases must always be accepted by HiGHS)",
        fwd.basis_consistency_failures
    );

    #[cfg(feature = "highs")]
    {
        let observed_simplex = fwd.simplex_iterations;
        assert!(
            (LO_SIMPLEX..=HI_SIMPLEX).contains(&observed_simplex),
            "basis_reconstruct_churn: simplex_iterations={observed_simplex} is outside \
             the ±5 % band [{LO_SIMPLEX}, {HI_SIMPLEX}] around pin={PINNED_SIMPLEX_ITERS}"
        );
    }

    // final_lb is deterministic for a fixed seed and scenario count; its last ULP
    // shifts with the HiGHS build, so pin to a relative tolerance that catches
    // real drift (~9 significant figures) while absorbing float noise.
    #[cfg(feature = "highs")]
    {
        let lb_rel_err = (result.final_lb - PINNED_FINAL_LB).abs() / PINNED_FINAL_LB.abs();
        assert!(
            lb_rel_err <= 1e-9,
            "basis_reconstruct_churn: final_lb={} deviates from pin={PINNED_FINAL_LB:.15e} \
             by relative error {lb_rel_err:.3e} (> 1e-9); the lower bound must be \
             deterministic for a fixed seed and scenario count",
            result.final_lb
        );
    }
}

// ---------------------------------------------------------------------------
// Test 2: No-churn — happy path, reconstruction active across iterations
// ---------------------------------------------------------------------------

/// No-churn happy path: cuts only accumulate. `basis_consistency_failures == 0`
/// confirms the always-baked reconstruction stays active across iterations.
#[test]
fn test_basis_reconstruct_no_churn_full_preservation() {
    let case_dir = d03_case_dir();
    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");

    config.training.forward_passes = Some(2);
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit { limit: 3 }]);

    config.training.cut_selection.selection = None;
    config.training.cut_selection.max_active_per_stage = None;

    let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
    let prepare_result =
        prepare_stochastic(system, &case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, &case_dir, false).expect("prepare_hydro_models must succeed");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup must build");

    let comm = StubComm;
    let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

    let outcome = setup
        .train(&mut solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "test_basis_reconstruct_no_churn_full_preservation: unexpected training error: {:?}",
        outcome.error
    );
    let result = outcome.result;
    assert_eq!(
        result.iterations, 3,
        "no_churn: expected 3 iterations, got {}",
        result.iterations
    );

    let fwd = sum_forward_deltas(&result.solver_stats_log);

    // basis_reconstructions is excluded from forward-log MPI packing (always 0
    // here); basis_consistency_failures == 0 is the proxy for "reconstruction is
    // active and producing valid bases".
    assert_eq!(
        fwd.basis_consistency_failures, 0,
        "no_churn: expected 0 basis rejections, got {}",
        fwd.basis_consistency_failures
    );

    assert!(
        result.final_lb.is_finite() && result.final_lb > 0.0,
        "no_churn: expected finite positive lower bound, got {}",
        result.final_lb
    );
}

// ---------------------------------------------------------------------------
// Test 3: Full churn — all iteration-1 cuts deactivated before iteration 2
// ---------------------------------------------------------------------------

/// Full-churn edge case: every iteration-1 cut deactivated before the
/// iteration-2 forward pass, so the reconstruction path meets an empty FCF.
/// Phase 2 rebuilds a fresh cold-start `setup2` with no stored basis, so
/// `basis_reconstructions == 0` is expected and the classification assertion is
/// dropped on purpose; the safety goal is that an empty FCF neither panics nor
/// produces an invalid basis (`basis_consistency_failures == 0`).
#[test]
fn test_basis_reconstruct_full_churn_no_rows_preserved() {
    let case_dir = d01_case_dir();
    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");

    // Two IterationLimit rules in "any" mode: limit 2 sizes the FCF pool to
    // (2+1) × forward_passes slots/stage (capacity headroom); limit 1 fires first
    // and stops after iteration 1.
    config.training.forward_passes = Some(2);
    config.training.stopping_rules = Some(vec![
        StoppingRuleConfig::IterationLimit { limit: 2 }, // FCF capacity sizing
        StoppingRuleConfig::IterationLimit { limit: 1 }, // actual stop point
    ]);
    let system = cobre_io::load_case(&case_dir).expect("load_case phase1 must succeed");
    let prepare_result =
        prepare_stochastic(system, &case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic phase1 must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, &case_dir, false).expect("prepare_hydro_models must succeed");

    let mut setup = StudySetup::new(&system, &config, stochastic, hydro_models)
        .expect("StudySetup phase1 must build");

    let comm = StubComm;
    let mut solver1 = ActiveSolver::new().expect("ActiveSolver phase1 must succeed");

    let outcome1 = setup
        .train(&mut solver1, &comm, 1, ActiveSolver::new, None, None)
        .expect("phase1 train must return Ok");
    assert!(
        outcome1.error.is_none(),
        "full_churn: phase 1 unexpected error: {:?}",
        outcome1.error
    );
    assert_eq!(
        outcome1.result.iterations, 1,
        "full_churn: phase 1 must complete exactly 1 iteration"
    );

    let cuts_after_iter1: Vec<usize> = setup
        .fcf
        .pools
        .iter()
        .map(cobre_sddp::cut::CutPool::active_count)
        .collect();
    let total_cuts_iter1: usize = cuts_after_iter1.iter().sum();
    assert!(
        total_cuts_iter1 > 0,
        "full_churn: phase 1 must generate at least 1 cut, got 0; \
         test cannot exercise the deactivation path otherwise"
    );

    {
        let fcf = &mut setup.fcf;
        for (stage, pool) in fcf.pools.iter_mut().enumerate() {
            let active_indices: Vec<u32> = (0..pool.populated_count)
                .filter(|&i| pool.active[i])
                .map(|i| i as u32)
                .collect();
            if !active_indices.is_empty() {
                pool.deactivate(&active_indices);
            }
            assert_eq!(
                pool.active_count(),
                0,
                "full_churn: stage {stage} pool must have 0 active cuts after deactivation, \
                 got {}",
                pool.active_count()
            );
        }
    }

    {
        config.training.stopping_rules =
            Some(vec![StoppingRuleConfig::IterationLimit { limit: 2 }]);
    }

    setup.set_start_iteration(1);

    let mut solver2 = ActiveSolver::new().expect("ActiveSolver phase2 must succeed");

    // The config is baked into setup at construction, so phase 2 cannot reuse the
    // phase-1 setup with the new stopping rule: rebuild a fresh setup from the same
    // system and transplant the fully-deactivated FCF into it.
    {
        let system2 = cobre_io::load_case(&case_dir).expect("load_case phase2 must succeed");
        let prepare2 =
            prepare_stochastic(system2, &case_dir, &config, 42, &ScenarioSource::default())
                .expect("prepare_stochastic phase2 must succeed");
        let system2 = prepare2.system;
        let stochastic2 = prepare2.stochastic;
        let hydro2 =
            prepare_hydro_models(&system2, &case_dir, false).expect("prepare_hydro_models phase2");

        let mut setup2 =
            StudySetup::new(&system2, &config, stochastic2, hydro2).expect("StudySetup phase2");

        // Read the placeholder-FCF metadata before the mutable borrow below
        // (borrow checker).
        let n_stages = setup.fcf.pools.len();
        let state_dim = setup.fcf.state_dimension;
        let fwd_passes = setup.loop_params.forward_passes;
        let max_iters = setup.loop_params.max_iterations;
        let placeholder_fcf = cobre_sddp::FutureCostFunction::new(
            n_stages,
            state_dim,
            fwd_passes,
            max_iters,
            &vec![0u32; n_stages],
        );
        let deactivated_fcf = std::mem::replace(&mut setup.fcf, placeholder_fcf);
        setup2.replace_fcf(deactivated_fcf);
        setup2.set_start_iteration(1);

        let outcome2 = setup2
            .train(&mut solver2, &comm, 1, ActiveSolver::new, None, None)
            .expect("phase2 train must return Ok");
        assert!(
            outcome2.error.is_none(),
            "full_churn: phase 2 unexpected error: {:?}",
            outcome2.error
        );
        assert_eq!(
            outcome2.result.iterations, 2,
            "full_churn: phase 2 must report 2 total iterations, got {}",
            outcome2.result.iterations
        );

        let result2 = outcome2.result;

        let iter2_fwd: Vec<&SolverStatsDelta> = result2
            .solver_stats_log
            .iter()
            .filter(|e| e.iteration == 2 && e.phase == "forward")
            .map(|e| &e.delta)
            .collect();

        assert!(
            !iter2_fwd.is_empty(),
            "full_churn: must have at least one forward log entry for iteration 2"
        );

        let iter2_fwd_agg = SolverStatsDelta::aggregate(iter2_fwd.into_iter());

        assert_eq!(
            iter2_fwd_agg.basis_consistency_failures, 0,
            "full_churn: iteration 2 forward must have 0 basis rejections, got {} \
             (empty FCF must not cause HiGHS to reject the warm-start basis)",
            iter2_fwd_agg.basis_consistency_failures
        );

        assert!(
            result2.final_lb.is_finite(),
            "full_churn: final_lb must be finite after full-churn iteration, got {}",
            result2.final_lb
        );
    }
}

// ---------------------------------------------------------------------------
// Test 4: Simulation smoke — baked-path simulate completes with zero failures
// ---------------------------------------------------------------------------

/// Smoke test: after 2-iteration training on D03, baked-path simulation with the
/// trained `basis_cache` reconstructs a basis per stage/scenario with zero
/// `basis_consistency_failures` (no reconstructed basis is rejected by `HiGHS`).
#[test]
fn simulate_baked_path_zero_consistency_failures() {
    let case_dir = d03_case_dir();
    let config_path = case_dir.join("config.json");
    let mut config = cobre_io::parse_config(&config_path).expect("config must parse");

    // 2 iterations so iter 2's forward pass captures a basis with non-empty
    // cut_row_slots into basis_cache for the simulation warm-start.
    config.training.forward_passes = Some(2);
    config.training.stopping_rules = Some(vec![StoppingRuleConfig::IterationLimit { limit: 2 }]);

    config.training.cut_selection.selection = None;
    config.training.cut_selection.max_active_per_stage = None;

    config.simulation.enabled = true;
    config.simulation.num_scenarios = 2;

    let system = cobre_io::load_case(&case_dir).expect("load_case must succeed");
    let prepare_result =
        prepare_stochastic(system, &case_dir, &config, 42, &ScenarioSource::default())
            .expect("prepare_stochastic must succeed");
    let system = prepare_result.system;
    let stochastic = prepare_result.stochastic;

    let hydro_models =
        prepare_hydro_models(&system, &case_dir, false).expect("prepare_hydro_models must succeed");

    let mut setup =
        StudySetup::new(&system, &config, stochastic, hydro_models).expect("StudySetup must build");

    let comm = StubComm;
    let mut train_solver = ActiveSolver::new().expect("ActiveSolver for training must succeed");

    let outcome = setup
        .train(&mut train_solver, &comm, 1, ActiveSolver::new, None, None)
        .expect("train must return Ok");
    assert!(
        outcome.error.is_none(),
        "simulate_warm_start: expected no training error, got: {:?}",
        outcome.error
    );
    assert_eq!(
        outcome.result.iterations, 2,
        "simulate_warm_start: expected 2 iterations, got {}",
        outcome.result.iterations
    );

    let baked_templates =
        outcome.result.baked_templates.as_deref().expect(
            "simulate_warm_start: training must produce baked_templates after >= 2 iterations",
        );
    let basis_cache = &outcome.result.basis_cache;
    assert!(
        basis_cache
            .iter()
            .any(|cb| cb.as_ref().is_some_and(|b| !b.cut_row_slots.is_empty())),
        "simulate_warm_start: at least one stage must have a CapturedBasis with \
         non-empty cut_row_slots in basis_cache after 2 iterations"
    );

    let mut pool = setup
        .create_workspace_pool(&comm, 1, ActiveSolver::new)
        .expect("simulation workspace pool must build");

    let io_capacity = setup.simulation_config.io_channel_capacity.max(1);
    let (result_tx, result_rx) = mpsc::sync_channel(io_capacity);

    // Drain results on a background thread to avoid channel backpressure.
    let drain_handle = std::thread::spawn(move || result_rx.into_iter().count());

    let sim_result = setup
        .simulate(
            &mut pool.workspaces,
            &comm,
            &result_tx,
            None,
            Some(baked_templates),
            basis_cache,
        )
        .expect("simulate must return Ok");

    drop(result_tx);
    drain_handle.join().expect("drain thread must not panic");

    let total_rejections: u64 = sim_result
        .solver_stats
        .iter()
        .map(|(_, _, delta)| delta.basis_consistency_failures)
        .sum();

    assert_eq!(
        total_rejections, 0,
        "simulate_warm_start: expected 0 basis_consistency_failures in baked-path simulation, \
         got {total_rejections} (reconstructed bases must always be accepted by HiGHS)"
    );
}
