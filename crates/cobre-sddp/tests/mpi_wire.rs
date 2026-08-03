//! Consolidated MPI wire-format integration tests for `cobre-sddp`.
//!
//! Each source domain lives in its own inner `mod` so the suite links the
//! statically-bound solver once rather than once per file. Per-`mod` scoping
//! isolates each group's helpers and fixtures. These exercise the basis/cut
//! broadcast wire format and the hydro-models output path through the public
//! API with no MPI spawn.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]

mod common;

mod test_mpi_hydro_models_output_path {
    //! Integration test: for `source: "computed"` hydros, `prepare_hydro_models`
    //! populates `fpha_export_rows` in memory and writes no file to disk.

    use std::path::Path;

    use cobre_sddp::prepare_hydro_models;

    fn d07_case_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("crates/<crate> must have a parent")
            .parent()
            .expect("crates/ must have a parent (repo root)")
            .join("examples/deterministic/d07-fpha-computed")
    }

    #[test]
    fn prepare_hydro_models_returns_fpha_rows_without_writing_files() {
        let case_dir = d07_case_dir();
        assert!(
            case_dir.exists(),
            "d07-fpha-computed fixture must exist at {case_dir:?}"
        );

        let system = cobre_io::load_case(&case_dir).expect("load_case must succeed on d07");

        let result = prepare_hydro_models(&system, &case_dir, false)
            .expect("prepare_hydro_models must succeed");

        assert!(
            !result.fpha_export_rows.is_empty(),
            "fpha_export_rows must be non-empty for a computed-source FPHA case; \
             got {} rows",
            result.fpha_export_rows.len()
        );
    }
}

mod test_mpi_basis_broadcast_large_len {
    //! Integration test: oversized MPI broadcast buffer is rejected.
    //!
    //! `CommError::InvalidBufferSize` must correctly round-trip through
    //! `SddpError::Communication` with both actual (oversized) and expected
    //! (`i32::MAX`) length fields preserved, enabling diagnostic messages.

    use cobre_comm::CommError;
    use cobre_sddp::SddpError;

    /// Mirrors the error `checked_broadcast_len` produces for
    /// `len = (i32::MAX as usize) + 1` — the smallest value that cannot be
    /// represented as an MPI count.
    #[test]
    fn comm_error_invalid_buffer_size_roundtrip() {
        let actual = (i32::MAX as usize) + 1;
        let expected = i32::MAX as usize;
        let operation = "broadcast_basis_cache_i32";

        let comm_err = CommError::InvalidBufferSize {
            operation,
            expected,
            actual,
        };
        let sddp_err = SddpError::Communication(comm_err);

        match sddp_err {
            SddpError::Communication(CommError::InvalidBufferSize {
                operation: op,
                expected: exp,
                actual: act,
            }) => {
                assert_eq!(
                    act,
                    (i32::MAX as usize) + 1,
                    "actual must equal (i32::MAX as usize) + 1"
                );
                assert_eq!(
                    exp,
                    i32::MAX as usize,
                    "expected must equal i32::MAX as usize"
                );
                assert_eq!(
                    op, "broadcast_basis_cache_i32",
                    "operation string must match"
                );
            }
            other => panic!(
                "expected SddpError::Communication(CommError::InvalidBufferSize {{ .. }}), got: \
                 {other:?}"
            ),
        }
    }

    #[test]
    fn comm_error_invalid_buffer_size_display_contains_counts() {
        let actual = (i32::MAX as usize) + 1;
        let expected = i32::MAX as usize;

        let err = CommError::InvalidBufferSize {
            operation: "broadcast_basis_cache_i32",
            expected,
            actual,
        };
        let msg = err.to_string();

        assert!(
            msg.contains(&actual.to_string()),
            "Display must contain actual count {actual}; got: {msg}"
        );
        assert!(
            msg.contains(&expected.to_string()),
            "Display must contain expected count {expected}; got: {msg}"
        );
    }
}

mod test_mpi_wire_format_version {
    //! MPI wire-format version guards: `CapturedBasis::try_from_broadcast_payload`
    //! and `deserialize_cut` must return `SddpError::Validation` on a corrupted
    //! version byte/field — a stale version must be rejected, never silently
    //! decoded as corrupt data. Exercised through the public API, no MPI spawn.

    use cobre_sddp::{
        SddpError,
        cut::wire::{CUT_WIRE_VERSION, cut_wire_size, deserialize_cut, serialize_cut},
        workspace::{BASIS_BROADCAST_WIRE_VERSION, CapturedBasis},
    };
    use cobre_solver::BasisStatus;

    // ---------------------------------------------------------------------------
    // basis wire-format version guard
    // ---------------------------------------------------------------------------

    #[test]
    fn basis_try_from_broadcast_payload_rejects_wrong_version() {
        let num_cols = 2;
        let num_rows = 3;
        let base_row_count = 1;
        let cut_slot_capacity = 2;
        let n_state = 1;

        let mut original = CapturedBasis::new(
            num_cols,
            num_rows,
            base_row_count,
            cut_slot_capacity,
            n_state,
            0,
        );
        // Minimal valid data so the Some-path sentinel (i32_buf[0] == 1) is written.
        original
            .basis
            .col_status
            .extend_from_slice(&[BasisStatus::Lower, BasisStatus::Basic]);
        original.basis.row_status.extend_from_slice(&[
            BasisStatus::Lower,
            BasisStatus::Basic,
            BasisStatus::Lower,
        ]);
        original.cut_row_slots.push(0_u32);
        original.cut_row_slots.push(1_u32);
        original.state_at_capture.push(42.0_f64);

        let mut i32_buf: Vec<i32> = Vec::new();
        let mut f64_buf: Vec<f64> = Vec::new();
        original.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

        // i32_buf layout (owned by to_broadcast_payload): [sentinel, version, ...] —
        // the version field is at index 1, which this test corrupts below.
        assert_eq!(i32_buf[0], 1, "sentinel must be 1 (Some path)");
        assert_eq!(
            i32_buf[1], BASIS_BROADCAST_WIRE_VERSION,
            "version field must equal BASIS_BROADCAST_WIRE_VERSION before corruption"
        );

        i32_buf[1] = 2;

        let mut i32_cursor = 0_usize;
        let mut f64_cursor = 0_usize;
        let result = CapturedBasis::try_from_broadcast_payload(
            0,
            &i32_buf,
            &mut i32_cursor,
            &f64_buf,
            &mut f64_cursor,
        );

        match result {
            Err(SddpError::Validation(ref msg)) => {
                assert!(
                    msg.contains("unsupported wire version 2"),
                    "error must contain 'unsupported wire version 2'; got: {msg}"
                );
            }
            other => panic!(
                "expected Err(SddpError::Validation(_)) containing 'unsupported wire version 2', \
                 got: {other:?}"
            ),
        }
    }

    // ---------------------------------------------------------------------------
    // Cut wire-format version guard
    // ---------------------------------------------------------------------------

    #[test]
    fn deserialize_cut_rejects_wrong_version() {
        let n_state = 2;
        let record_size = cut_wire_size(n_state);
        let mut buf = vec![0u8; record_size];

        serialize_cut(
            &mut buf,
            /* slot_index */ 0,
            /* iteration */ 1,
            /* forward_pass_index */ 0,
            /* intercept */ 99.0,
            /* coefficients */ &[1.0, 2.0],
        );

        assert_eq!(
            buf[0], CUT_WIRE_VERSION,
            "serialize_cut must write the current version byte"
        );

        buf[0] = 99_u8;

        let result = deserialize_cut(&buf, n_state);

        match result {
            Err(SddpError::Validation(ref msg)) => {
                assert!(
                    msg.contains("99"),
                    "error must reference the corrupted version 99; got: {msg}"
                );
            }
            other => panic!(
                "expected Err(SddpError::Validation(_)) referencing the corrupted version, \
                 got: {other:?}"
            ),
        }
    }
}

mod test_mpi_4rank_basis_broadcast_round_trip {
    //! `to_broadcast_payload` / `try_from_broadcast_payload` produce bit-identical
    //! `CapturedBasis` values across four ranks reading from the same shared buffers.

    use cobre_sddp::workspace::{BASIS_BROADCAST_WIRE_VERSION, CapturedBasis};
    use cobre_solver::BasisStatus;

    /// The seven canonical statuses, indexed by `(seed + i) % 7` in
    /// `make_captured_basis` to derive a deterministic, seed-varying sequence.
    const BASIS_VARIANTS: [BasisStatus; 7] = [
        BasisStatus::Lower,
        BasisStatus::Basic,
        BasisStatus::Upper,
        BasisStatus::Zero,
        BasisStatus::Nonbasic,
        BasisStatus::Superbasic,
        BasisStatus::Fixed,
    ];

    fn make_captured_basis(seed: u32) -> CapturedBasis {
        let num_cols = 4_usize;
        let num_rows = 6_usize;
        let base_row_count = 2_usize;
        let cut_slot_capacity = 4_usize;
        let n_state = 3_usize;

        let mut cb = CapturedBasis::new(
            num_cols,
            num_rows,
            base_row_count,
            cut_slot_capacity,
            n_state,
            0,
        );

        for i in 0..num_cols {
            let idx = (seed as usize + i) % BASIS_VARIANTS.len();
            cb.basis.col_status.push(BASIS_VARIANTS[idx]);
        }
        for i in 0..num_rows {
            let idx = (seed as usize * 2 + i) % BASIS_VARIANTS.len();
            cb.basis.row_status.push(BASIS_VARIANTS[idx]);
        }
        for i in 0..cut_slot_capacity {
            cb.cut_row_slots.push(seed + i as u32);
        }
        for i in 0..n_state {
            cb.state_at_capture.push(f64::from(seed) + (i as f64) * 0.5);
        }

        cb
    }

    fn assert_captured_basis_eq(a: &CapturedBasis, b: &CapturedBasis, label: &str) {
        assert_eq!(
            a.basis.col_status, b.basis.col_status,
            "{label}: col_status mismatch"
        );
        assert_eq!(
            a.basis.row_status, b.basis.row_status,
            "{label}: row_status mismatch"
        );
        assert_eq!(
            a.base_row_count, b.base_row_count,
            "{label}: base_row_count mismatch"
        );
        assert_eq!(
            a.cut_row_slots, b.cut_row_slots,
            "{label}: cut_row_slots mismatch"
        );
        assert_eq!(
            a.state_at_capture, b.state_at_capture,
            "{label}: state_at_capture mismatch"
        );
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn four_rank_basis_broadcast_round_trip() {
        const NUM_RANKS: usize = 4;
        const NUM_STAGES: usize = 3;

        let stage0_basis = make_captured_basis(10);
        let stage2_basis = make_captured_basis(20);

        let mut i32_buf: Vec<i32> = Vec::new();
        let mut f64_buf: Vec<f64> = Vec::new();

        stage0_basis.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

        // None is encoded as the bare `0_i32` sentinel, not by absence.
        i32_buf.push(0_i32);

        stage2_basis.to_broadcast_payload(&mut i32_buf, &mut f64_buf);

        // A Some stage's i32 layout: [1 (sentinel), VERSION, col_len, row_len,
        // base_row_count, cut_slot_count, state_len, ...].
        assert_eq!(i32_buf[0], 1, "stage 0 sentinel must be 1");
        assert_eq!(
            i32_buf[1], BASIS_BROADCAST_WIRE_VERSION,
            "stage 0 version must equal BASIS_BROADCAST_WIRE_VERSION"
        );

        // Each rank reads independently from the same shared buffers.
        let unpack_all_stages = |rank: usize| -> Vec<Option<CapturedBasis>> {
            let mut i32_cursor = 0_usize;
            let mut f64_cursor = 0_usize;
            let mut stages = Vec::with_capacity(NUM_STAGES);
            for stage in 0..NUM_STAGES {
                let result = CapturedBasis::try_from_broadcast_payload(
                    stage,
                    &i32_buf,
                    &mut i32_cursor,
                    &f64_buf,
                    &mut f64_cursor,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "rank {rank}: try_from_broadcast_payload must succeed at stage {stage}: {e}"
                    )
                });
                stages.push(result);
            }
            stages
        };

        let results: Vec<Vec<Option<CapturedBasis>>> =
            (0..NUM_RANKS).map(unpack_all_stages).collect();

        let ref_stage0 = results[0][0].as_ref().expect("rank 0 stage 0 must be Some");
        let ref_stage2 = results[0][2].as_ref().expect("rank 0 stage 2 must be Some");
        assert!(results[0][1].is_none(), "rank 0 stage 1 must be None");

        for (rank, rank_results) in results.iter().enumerate().skip(1) {
            let other_stage0 = rank_results[0]
                .as_ref()
                .unwrap_or_else(|| panic!("rank {rank} stage 0 must be Some"));
            assert_captured_basis_eq(ref_stage0, other_stage0, &format!("rank {rank} stage 0"));

            assert!(
                rank_results[1].is_none(),
                "rank {rank} stage 1 must be None"
            );

            let other_stage2 = rank_results[2]
                .as_ref()
                .unwrap_or_else(|| panic!("rank {rank} stage 2 must be Some"));
            assert_captured_basis_eq(ref_stage2, other_stage2, &format!("rank {rank} stage 2"));
        }

        assert_captured_basis_eq(&stage0_basis, ref_stage0, "pack/unpack parity stage 0");
        assert_captured_basis_eq(&stage2_basis, ref_stage2, "pack/unpack parity stage 2");
    }
}

#[cfg(all(feature = "highs", feature = "test-support"))]
mod retry_armed_determinism {
    //! Retry-armed determinism gate: the tuned backward profile (`SteepestEdge`
    //! dual edge weight, Curtis-Reid scaling, `Row` pricing, primal tolerance
    //! `1e-7`) with a deliberately low `simplex_iteration_limit` forces the
    //! `HiGHS` retry escalation ladder
    //! (`crates/cobre-solver/src/backends/highs/retry.rs`) to fire, then
    //! asserts the final training lower bound is bitwise
    //! identical across four execution shapes of the SAME config: threads=k,
    //! threads=1, a same-shape repeat, and a faithful 2-rank leg. Runs on both
    //! an expectation and a `CVaR` risk configuration of the same fixture.
    //!
    //! Power statement: this gate catches a profile option dropped at the
    //! retry-finalization seam (`reapply_profile` after
    //! `restore_default_settings`) that manifests only once a solve genuinely
    //! retries; it has no power on a case/length that produces zero retries.
    //! `total_retries > 0` is asserted per shape as the gate's own
    //! self-check, not an incidental fact. `forced_retry_profiles` arms every
    //! new profile field this gate can reach — including `use_warm_start`,
    //! which the d03 fixture tolerates without ever driving a shape's
    //! `total_retries` to zero — at a non-default in-range value, so the
    //! four-shape bitwise `final_lb` comparison also proves each survives the
    //! seam at integration scope; this complements the unit-level
    //! `full_profile_survives_retry_finalization_seam` readback in
    //! `crates/cobre-solver/tests/profile_retry_composition.rs`.

    use std::path::Path;

    use cobre_comm::Communicator;
    use cobre_io::config::{
        BackwardScheduler, DualEdgeWeight, PhaseSolverProfileConfig, PresolveMode, PriceStrategy,
        ScaleStrategy,
    };
    use cobre_sddp::{Phase, RiskMeasure, SolverProfiles, StudySetup};
    use cobre_solver::ActiveSolver;

    use crate::common::{Rank0Of2, StubComm};

    /// Low enough that the tuned backward profile's first attempt cannot
    /// finish within the cap on every stage solve of the d03 fixture, arming
    /// the retry-escalation ladder on every solve rather than occasionally;
    /// tuned empirically against this fixture, not derived from a closed
    /// form.
    const FORCED_SIMPLEX_ITERATION_LIMIT: u32 = 1;

    /// The tuned backward profile (`SteepestEdge` / Curtis-Reid / Row / ptol
    /// `1e-7`) forced to [`FORCED_SIMPLEX_ITERATION_LIMIT`]. `forward` stays
    /// byte-neutral (`Phase::Forward.resolve_profile(None)`): only the
    /// backward-pass retry-finalization seam is under test.
    fn forced_retry_profiles() -> SolverProfiles {
        let tuned = PhaseSolverProfileConfig {
            dual_edge_weight: Some(DualEdgeWeight::SteepestEdge),
            scale: Some(ScaleStrategy::SolverScaling),
            price: Some(PriceStrategy::Row),
            primal_feasibility_tolerance: Some(1e-7),
            dual_feasibility_tolerance: Some(1e-8),
            presolve: Some(PresolveMode::Off),
            simplex_update_limit: Some(1000),
            cost_perturbation: Some(1.0),
            refactor_error_tolerance: Some(1e-5),
            factor_pivot_threshold: Some(0.2),
            use_warm_start: Some(false),
            steepest_edge_devex_fallback_threshold: Some(20.0),
        };
        let mut backward = Phase::Backward.resolve_profile(Some(&tuned));
        backward.simplex_iteration_limit = FORCED_SIMPLEX_ITERATION_LIMIT;
        SolverProfiles {
            forward: Phase::Forward.resolve_profile(None),
            backward,
            backward_scheduler: BackwardScheduler::default(),
            hardest_first_claim_order: true,
        }
    }

    fn d03_case_dir() -> &'static Path {
        Path::new("../../examples/deterministic/d03-two-hydro-cascade")
    }

    /// Thin wrapper over [`crate::common::fresh_setup_with`] (no config
    /// mutation) — see its doc for the construction pipeline. Each shape
    /// trains from an independently-built setup, never a warm-started reuse,
    /// so the four shapes compare cold-to-cold.
    fn fresh_setup(case_dir: &Path) -> StudySetup {
        crate::common::fresh_setup_with(case_dir, |_| {})
    }

    /// Train one shape on a fresh [`StudySetup`], returning
    /// `(final_lb, total_retries)`. `total_retries` sums
    /// `solver_stats_log`'s per-iteration, per-phase deltas — the run's
    /// reduced `SolverStatistics.retry_count`, already aggregated across every
    /// workspace/rank the phase distributed work to.
    fn run_shape(
        case_dir: &Path,
        n_threads: usize,
        comm: &impl Communicator,
        risk_measures: Option<Vec<RiskMeasure>>,
    ) -> (f64, u64) {
        let mut setup = fresh_setup(case_dir);
        if let Some(risk_measures) = risk_measures {
            setup.set_risk_measures(risk_measures);
        }
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

        let outcome = setup
            .train_with_solver_profiles(
                &mut solver,
                comm,
                n_threads,
                ActiveSolver::new,
                forced_retry_profiles(),
            )
            .expect("train_with_solver_profiles must return Ok");
        assert!(
            outcome.error.is_none(),
            "expected no training error, got: {:?}",
            outcome.error
        );

        let total_retries: u64 = outcome
            .result
            .solver_stats_log
            .iter()
            .map(|entry| entry.delta.retry_attempts)
            .sum();

        (outcome.result.final_lb, total_retries)
    }

    /// Assert bitwise-identical `final_lb` across the four shapes of one
    /// config, and that every shape's reduced retry count is `> 0` — a
    /// zero-retry shape is a powerless gate, not a pass.
    fn assert_shapes_agree(case_dir: &Path, risk_measures: Option<Vec<RiskMeasure>>) {
        let stub1 = StubComm;
        let rank0_of_2 = Rank0Of2;

        let (lb_threads_k, retries_k) = run_shape(case_dir, 4, &stub1, risk_measures.clone());
        let (lb_threads_1, retries_1) = run_shape(case_dir, 1, &stub1, risk_measures.clone());
        let (lb_repeat, retries_repeat) = run_shape(case_dir, 1, &stub1, risk_measures.clone());
        let (lb_2rank, retries_2rank) = run_shape(case_dir, 1, &rank0_of_2, risk_measures);

        for (label, retries) in [
            ("threads=4", retries_k),
            ("threads=1", retries_1),
            ("same-shape repeat", retries_repeat),
            ("2-rank", retries_2rank),
        ] {
            assert!(
                retries > 0,
                "{label}: forced-retry profile produced 0 retries on this fixture — the gate \
                 is powerless here; tighten FORCED_SIMPLEX_ITERATION_LIMIT"
            );
        }

        assert_eq!(
            lb_threads_k.to_bits(),
            lb_threads_1.to_bits(),
            "threads=4 vs threads=1 final lower bound must be bitwise identical"
        );
        assert_eq!(
            lb_threads_1.to_bits(),
            lb_repeat.to_bits(),
            "same-shape repeat final lower bound must be bitwise identical"
        );
        assert_eq!(
            lb_threads_1.to_bits(),
            lb_2rank.to_bits(),
            "2-rank final lower bound must be bitwise identical"
        );
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn retry_armed_determinism_expectation() {
        assert_shapes_agree(d03_case_dir(), None);
    }

    /// `alpha=0.5, lambda=1.0` mirror `conformance.rs`'s
    /// `cvar_alpha_half_concentrates_on_worst` fixture.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn retry_armed_determinism_cvar() {
        let case_dir = d03_case_dir();
        let num_stages = fresh_setup(case_dir).num_stages();
        let risk_measures = vec![
            RiskMeasure::CVaR {
                alpha: 0.5,
                lambda: 1.0
            };
            num_stages
        ];
        assert_shapes_agree(case_dir, Some(risk_measures));
    }
}

#[cfg(all(feature = "highs", feature = "test-support"))]
mod opening_order_determinism {
    //! Shortest-chain opening-order determinism gate: trains `examples/1dtoy`
    //! (a stochastic, multi-opening fixture; the shortest-chain solve order is
    //! intrinsic) via the public `train` entry point — config-resolved
    //! (default) solver profiles, no forced retry — and asserts the final
    //! training lower bound is bitwise identical across four execution shapes
    //! of the SAME config: threads=k, threads=1, a same-shape repeat, and a
    //! faithful 2-rank leg.
    //!
    //! Power statement: this gate has no power on a fixture whose solve chain
    //! has nothing to reorder (the shortest-chain order no-ops below 3
    //! openings per stage, see `noise_key::apply_chain_order`);
    //! `n_openings >= 3` on at least one stage is asserted as the gate's own
    //! self-check, not an incidental fact.
    //!
    //! The real multi-rank leg is the existing MPI SLURM Integration job
    //! (`.github/workflows/mpi-slurm.yml`, `tests/slurm/run-tests.sh`), which
    //! compares `mpiexec -n 1` against `-n 2` on `examples/4ree` bit-for-bit;
    //! this in-process gate does not reproduce that leg.

    use std::path::Path;

    use cobre_comm::Communicator;
    use cobre_sddp::StudySetup;
    use cobre_solver::ActiveSolver;

    use crate::common::{Rank0Of2, StubComm};

    fn fixture_case_dir() -> &'static Path {
        Path::new("../../examples/1dtoy")
    }

    /// Thin wrapper over [`crate::common::fresh_setup_with`] (no config
    /// mutation) — see its doc for the construction pipeline. Each shape
    /// trains from an independently-built setup, never a warm-started reuse,
    /// so the four shapes compare cold-to-cold.
    fn fresh_setup(case_dir: &Path) -> StudySetup {
        crate::common::fresh_setup_with(case_dir, |_| {})
    }

    /// Train one shape on a fresh [`StudySetup`] via the public `train` entry
    /// point (config-resolved default profiles), returning `final_lb`.
    fn run_shape(case_dir: &Path, n_threads: usize, comm: &impl Communicator) -> f64 {
        let mut setup = fresh_setup(case_dir);
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");

        let outcome = setup
            .train(&mut solver, comm, n_threads, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(
            outcome.error.is_none(),
            "expected no training error, got: {:?}",
            outcome.error
        );

        outcome.result.final_lb
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn opening_order_determinism() {
        let case_dir = fixture_case_dir();

        let probe = fresh_setup(case_dir);
        let tree_view = probe.stochastic.tree_view();
        let has_multi_opening_stage =
            (0..probe.num_stages()).any(|stage| tree_view.n_openings(stage) >= 3);
        assert!(
            has_multi_opening_stage,
            "opening_order_determinism: fixture {case_dir:?} has no stage with \
             n_openings >= 3 — the gate is powerless here (the shortest-chain order \
             is a no-op below 3 openings); point it at a genuinely multi-opening case"
        );

        let stub = StubComm;
        let rank0_of_2 = Rank0Of2;

        let lb_threads_k = run_shape(case_dir, 4, &stub);
        let lb_threads_1 = run_shape(case_dir, 1, &stub);
        let lb_repeat = run_shape(case_dir, 1, &stub);
        let lb_2rank = run_shape(case_dir, 1, &rank0_of_2);

        assert_eq!(
            lb_threads_k.to_bits(),
            lb_threads_1.to_bits(),
            "threads=4 vs threads=1 final lower bound must be bitwise identical"
        );
        assert_eq!(
            lb_threads_1.to_bits(),
            lb_repeat.to_bits(),
            "same-shape repeat final lower bound must be bitwise identical"
        );
        assert_eq!(
            lb_threads_1.to_bits(),
            lb_2rank.to_bits(),
            "2-rank final lower bound must be bitwise identical"
        );
    }
}

#[cfg(all(feature = "highs", feature = "test-support"))]
mod derived_inflow_seeds_rank_invariance {
    //! Rank-invariance gate for the derived PAR lag/accumulator seed: every
    //! rank derives `DerivedInflowSeeds` locally from the broadcast `System`
    //! inside `StudySetup::from_broadcast_params`, so a 1-rank run and a
    //! `Rank0Of2` 2-rank stub of the same fixture must seed a bit-identical
    //! stage-0 state vector and converge to a bit-identical final lower
    //! bound; a same-shape repeat must reproduce both as well.
    //!
    //! `examples/deterministic/d16-par1-lag-shift` is the fixture: single
    //! hydro, PAR(1), monthly, one opening per stage, and its
    //! `recent_observations` window is the sole source of the stage-0 lag-1
    //! seed (see the case's own pin in `deterministic.rs`), so the seed this
    //! gate exercises is non-trivial.

    use std::path::Path;

    use cobre_comm::Communicator;
    use cobre_solver::ActiveSolver;

    use crate::common::{Rank0Of2, StubComm};

    fn fixture_case_dir() -> &'static Path {
        Path::new("../../examples/deterministic/d16-par1-lag-shift")
    }

    /// Build a fresh `StudySetup` (see `crate::common::fresh_setup_with`'s doc
    /// for the construction pipeline — the same `from_broadcast_params` entry
    /// point every rank runs), capture the stage-0 seeded state vector, then
    /// train via the public `train` entry point and return
    /// `(state_bits, final_lb_bits)`.
    fn run_shape(case_dir: &Path, comm: &impl Communicator) -> (Vec<u64>, u64) {
        let mut setup = crate::common::fresh_setup_with(case_dir, |_| {});
        let state_bits: Vec<u64> = setup
            .training_ctx()
            .initial_state
            .iter()
            .map(|v| v.to_bits())
            .collect();

        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = setup
            .train(&mut solver, comm, 1, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(
            outcome.error.is_none(),
            "expected no training error, got: {:?}",
            outcome.error
        );

        (state_bits, outcome.result.final_lb.to_bits())
    }

    #[test]
    fn derived_inflow_seeds_rank_invariant() {
        let case_dir = fixture_case_dir();

        let probe = crate::common::fresh_setup_with(case_dir, |_| {});
        assert!(
            probe.training_ctx().initial_state.iter().any(|&v| v != 0.0),
            "derived_inflow_seeds_rank_invariant: fixture {case_dir:?} seeds an all-zero \
             stage-0 state vector — the gate is powerless here; point it at a fixture whose \
             recent_observations window feeds a non-trivial seed"
        );

        let stub = StubComm;
        let rank0_of_2 = Rank0Of2;

        let (state_1rank, lb_1rank) = run_shape(case_dir, &stub);
        let (state_1rank_repeat, lb_1rank_repeat) = run_shape(case_dir, &stub);
        let (state_2rank, lb_2rank) = run_shape(case_dir, &rank0_of_2);

        assert_eq!(
            state_1rank, state_1rank_repeat,
            "same-shape repeat: stage-0 seeded state vector must be bitwise identical"
        );
        assert_eq!(
            lb_1rank, lb_1rank_repeat,
            "same-shape repeat: final lower bound must be bitwise identical"
        );

        assert_eq!(
            state_1rank, state_2rank,
            "1-rank vs 2-rank stub: stage-0 seeded state vector must be bitwise identical"
        );
        assert_eq!(
            lb_1rank, lb_2rank,
            "1-rank vs 2-rank stub: final lower bound must be bitwise identical"
        );
    }
}

mod by_node_scheduler_determinism {
    //! by-node scheduler determinism gates: train `examples/1dtoy`
    //! under `training.parallelism.backward_scheduler` via the public
    //! `train` entry point. Hardest-first claim ordering is always-on under
    //! `by_node` (no config field gates it), so
    //! `by_node_scheduler_determinism_expectation` and `by_node_scheduler_determinism_cvar`
    //! exercise hardest-first-on from iteration 2 onward (`CVaR` coverage rides `_cvar`);
    //! both still assert `final_lb` is bitwise identical across five execution
    //! shapes of the SAME config — threads=4, a same-shape threads=4 repeat
    //! (the claim loop's run-to-run assignment randomization), threads=2,
    //! threads=1, and a `Rank0Of2` 2-rank stub at threads=4 — on both an
    //! expectation and a `CVaR` risk configuration.
    //! `hardest_first_claim_order_is_result_neutral` is the direct hardest-first-on-vs-off gate.
    //! `by_node_degenerates_on_single_opening` and
    //! `by_node_handles_non_uniform_cut_projection` are the two places a genuinely
    //! executed by-node run (`process_stage_backward_by_node`'s own claim loop, not the
    //! DCS bypass below) is compared directly against the by-scenario path's `final_lb`: the
    //! former on a single-opening deterministic case whose resolved
    //! opening-block count is `1` (a by-scenario-equivalent unit), the latter on a
    //! case whose per-stage cut-state projection dimension varies across
    //! stages. `by_node_falls_back_to_by_scenario_under_active_dcs` also compares
    //! two labeled runs, but both execute the SAME by-scenario code path
    //! under active DCS, so it pins the fallback dispatch rather than the by-node scheduler's
    //! own arithmetic; `by_node_generates_one_cut_per_trial_state` pins cut-count
    //! parity; `by_node_populates_backward_wall_ms` pins the telemetry surface.
    //! The scratch arena's no-alloc property is pinned primarily by
    //! `by_node_scratch`'s `by_node_scratch_capacity_stable_across_training`; the 5-way
    //! gates additionally reuse `by_node_scratch::run_by_node_one_iteration` as a
    //! defense-in-depth capacity check paired with the threads=4 leg.
    //!
    //! Power statement: the 5-way gates are powerless on a fixture whose
    //! resolved opening-block count never reaches `2` on any stage (a single
    //! block is a by-scenario-equivalent unit); each asserts this as its own
    //! self-check, mirroring `opening_order_determinism`'s `n_openings >= 3`
    //! check.
    //!
    //! A real multi-rank by-node run is not CI-covered; the in-process
    //! `Rank0Of2` 2-rank stub is the CI-time signal. The by-node
    //! scheduler is opt-in and the existing MPI SLURM Integration job trains
    //! the default scheduler on `examples/4ree`, not `by_node`.

    use std::path::Path;
    use std::sync::mpsc;

    use cobre_comm::Communicator;
    use cobre_core::{TrainingEvent, WorkerTimingPhase};
    use cobre_io::Config;
    use cobre_io::config::{BackwardScheduler, SelectionMethod, StoppingRuleConfig};
    use cobre_sddp::{RiskMeasure, StudySetup};
    use cobre_solver::ActiveSolver;

    use crate::by_node_scratch::run_by_node_one_iteration;
    use crate::common::{Rank0Of2, StubComm};

    fn fixture_case_dir() -> &'static Path {
        Path::new("../../examples/1dtoy")
    }

    /// Delegates to [`crate::common::fresh_setup_with`] — see its doc for the
    /// construction pipeline.
    fn fresh_setup_with(case_dir: &Path, mutate: impl FnOnce(&mut Config)) -> StudySetup {
        crate::common::fresh_setup_with(case_dir, mutate)
    }

    /// Build a fresh [`StudySetup`] with
    /// `training.parallelism.backward_scheduler` forced to `scheduler`.
    fn fresh_setup(case_dir: &Path, scheduler: BackwardScheduler) -> StudySetup {
        fresh_setup_with(case_dir, |config| {
            config.training.parallelism.backward_scheduler = scheduler;
        })
    }

    /// Like [`fresh_setup`], additionally forcing `cut_selection` to `Dynamic`
    /// active from iteration 1 — DCS is active on every solved iteration.
    fn fresh_setup_with_active_dcs(case_dir: &Path, scheduler: BackwardScheduler) -> StudySetup {
        fresh_setup_with(case_dir, |config| {
            config.training.parallelism.backward_scheduler = scheduler;
            config.training.cut_selection.selection = Some(SelectionMethod::Dynamic {
                start_iteration: 1,
                seed_window: 5,
                candidate_recency: None,
                max_added_per_round: 10,
                violation_tolerance: 1e-10,
            });
        })
    }

    /// Like [`fresh_setup`], additionally forcing exactly one training
    /// iteration (`training.stopping_rules = iteration_limit(1)`).
    fn fresh_setup_one_iteration(case_dir: &Path, scheduler: BackwardScheduler) -> StudySetup {
        fresh_setup_with(case_dir, |config| {
            config.training.parallelism.backward_scheduler = scheduler;
            config.training.stopping_rules =
                Some(vec![StoppingRuleConfig::IterationLimit { limit: 1 }]);
        })
    }

    /// Train `setup` via the public `train` entry point, returning `final_lb`.
    fn train_final_lb(mut setup: StudySetup, n_threads: usize, comm: &impl Communicator) -> f64 {
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = setup
            .train(&mut solver, comm, n_threads, ActiveSolver::new, None, None)
            .expect("train must return Ok");
        assert!(
            outcome.error.is_none(),
            "expected no training error, got: {:?}",
            outcome.error
        );
        outcome.result.final_lb
    }

    /// Train `setup` (one iteration) via the public `train` entry point,
    /// returning the single `BackwardPassComplete` event's `rows_generated` —
    /// `BackwardResult::cuts_generated` summed across all stages for that
    /// iteration.
    fn train_rows_generated(mut setup: StudySetup, comm: &impl Communicator) -> u32 {
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();
        let outcome = setup
            .train(
                &mut solver,
                comm,
                1,
                ActiveSolver::new,
                Some(event_tx),
                None,
            )
            .expect("train must return Ok");
        assert!(
            outcome.error.is_none(),
            "expected no training error, got: {:?}",
            outcome.error
        );

        let rows_generated: Vec<u32> = event_rx
            .try_iter()
            .filter_map(|e| {
                if let TrainingEvent::BackwardPassComplete { rows_generated, .. } = e {
                    Some(rows_generated)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            rows_generated.len(),
            1,
            "one-iteration training must emit exactly one BackwardPassComplete event, got {}",
            rows_generated.len()
        );
        rows_generated[0]
    }

    /// Resolved opening-block count for `n_openings` under the default
    /// (unconfigured) block size — mirrors `by_node::resolve_block_size`
    /// / `by_node::by_node_block_count` (both `pub(crate)`, unreachable
    /// from this external test crate).
    fn resolved_block_count(n_openings: usize) -> usize {
        let block_size = n_openings.div_ceil(2).min(n_openings);
        n_openings.div_ceil(block_size.max(1))
    }

    /// Powerless-gate self-check: a fixture whose every stage resolves to a
    /// single opening-block gives the opening-block scheduler nothing to distinguish from the trial-point path (see
    /// `by_node_degenerates_on_single_opening`), so at least one
    /// stage's resolved block count must reach `>= 2`.
    fn assert_has_multi_block_stage(case_dir: &Path) {
        let probe = fresh_setup(case_dir, BackwardScheduler::ByNode { block_size: None });
        let tree_view = probe.stochastic.tree_view();
        let has_multi_block_stage = (0..probe.num_stages())
            .any(|stage| resolved_block_count(tree_view.n_openings(stage)) >= 2);
        assert!(
            has_multi_block_stage,
            "by_node_scheduler_determinism: fixture {} has no stage whose resolved \
             opening-block count is >= 2 under the default block size — the gate is \
             powerless here; point it at a genuinely multi-opening case",
            case_dir.display()
        );
    }

    /// Defense-in-depth capacity check paired with the threads=4 leg below:
    /// the opening-block scratch arena's capacity (the no-alloc property `by_node_scratch`'s
    /// `by_node_scratch_capacity_stable_across_training` primarily pins) is
    /// reproduced identically across two independent direct-drive runs of the
    /// same fixture.
    fn assert_by_node_scratch_capacity_invariant() {
        const N_OPENINGS: usize = 4;
        let capacity_a = run_by_node_one_iteration(N_OPENINGS).by_node_scratch_arena_capacity();
        let capacity_b = run_by_node_one_iteration(N_OPENINGS).by_node_scratch_arena_capacity();
        assert_eq!(
            capacity_a, capacity_b,
            "opening-block scratch arena capacity must be reproducible across independent direct-drive runs"
        );
    }

    /// Train one shape (`by_node` forced, optional per-stage risk
    /// measures) via the public `train` entry point, returning `final_lb`.
    fn run_shape(
        case_dir: &Path,
        n_threads: usize,
        comm: &impl Communicator,
        risk_measures: Option<Vec<RiskMeasure>>,
    ) -> f64 {
        let mut setup = fresh_setup(case_dir, BackwardScheduler::ByNode { block_size: None });
        if let Some(risk_measures) = risk_measures {
            setup.set_risk_measures(risk_measures);
        }
        train_final_lb(setup, n_threads, comm)
    }

    /// Train the 5 execution shapes of one config on the SAME
    /// `by_node`-forced fixture and assert `final_lb.to_bits()` is
    /// bitwise identical across all five: threads=4, a same-shape threads=4
    /// repeat (the claim loop's run-to-run assignment randomization),
    /// threads=2, threads=1, and a `Rank0Of2` 2-rank stub at threads=4.
    fn assert_by_node_shapes_agree(case_dir: &Path, risk_measures: Option<Vec<RiskMeasure>>) {
        assert_has_multi_block_stage(case_dir);

        let stub = StubComm;
        let rank0_of_2 = Rank0Of2;

        let lb_threads_4 = run_shape(case_dir, 4, &stub, risk_measures.clone());
        assert_by_node_scratch_capacity_invariant();
        let lb_threads_4_repeat = run_shape(case_dir, 4, &stub, risk_measures.clone());
        let lb_threads_2 = run_shape(case_dir, 2, &stub, risk_measures.clone());
        let lb_threads_1 = run_shape(case_dir, 1, &stub, risk_measures.clone());
        let lb_2rank = run_shape(case_dir, 4, &rank0_of_2, risk_measures);

        for (label, bits) in [
            ("threads=4 same-shape repeat", lb_threads_4_repeat.to_bits()),
            ("threads=2", lb_threads_2.to_bits()),
            ("threads=1", lb_threads_1.to_bits()),
            ("2-rank stub (threads=4)", lb_2rank.to_bits()),
        ] {
            assert_eq!(
                lb_threads_4.to_bits(),
                bits,
                "{label} final lower bound must be bitwise identical to threads=4"
            );
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn by_node_scheduler_determinism_expectation() {
        assert_by_node_shapes_agree(fixture_case_dir(), None);
    }

    /// `alpha=0.5, lambda=1.0` mirrors `retry_armed_determinism_cvar`'s
    /// fixture — pins the canonical-ascending-m CVaR-safe aggregation under
    /// the opening-block scheduler's per-`(m, ω)` arena.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn by_node_scheduler_determinism_cvar() {
        let case_dir = fixture_case_dir();
        let num_stages =
            fresh_setup(case_dir, BackwardScheduler::ByNode { block_size: None }).num_stages();
        let risk_measures = vec![
            RiskMeasure::CVaR {
                alpha: 0.5,
                lambda: 1.0
            };
            num_stages
        ];
        assert_by_node_shapes_agree(case_dir, Some(risk_measures));
    }

    /// Hardest-first claim-order byte-neutrality gate (sddp.md "Opening-block
    /// scheduler is warm-start-only" — hardest-first result-neutrality): training the
    /// SAME `by_node`-forced fixture with hardest-first on (the production
    /// default) and again with `set_hardest_first_claim_order(false)` (the canonical
    /// ascending block order) must produce a bit-identical `final_lb` at the
    /// same thread count — hardest-first reorders only which worker claims which
    /// block, never the generated cut set.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn hardest_first_claim_order_is_result_neutral() {
        let case_dir = fixture_case_dir();
        assert_has_multi_block_stage(case_dir);
        let stub = StubComm;

        let lb_hardest_first_on = run_shape(case_dir, 4, &stub, None);

        let mut setup_hardest_first_off =
            fresh_setup(case_dir, BackwardScheduler::ByNode { block_size: None });
        setup_hardest_first_off.set_hardest_first_claim_order(false);
        let lb_hardest_first_off = train_final_lb(setup_hardest_first_off, 4, &stub);

        assert_eq!(
            lb_hardest_first_on.to_bits(),
            lb_hardest_first_off.to_bits(),
            "Hardest-first-on and -off (canonical claim order) must produce a bit-identical final \
             lower bound"
        );
    }

    fn single_opening_case_dir() -> &'static Path {
        Path::new("../../examples/deterministic/d01-thermal-dispatch")
    }

    /// Single-opening degeneracy gate: `d01-thermal-dispatch` has exactly one
    /// opening per stage, so `by_node`'s resolved block count is `1` —
    /// the whole trial point, a trial-point-equivalent unit — and `final_lb` must
    /// equal the `by_scenario` run bit-for-bit. The 5-way gates above compare
    /// opening-block-to-opening-block across shapes only; this and
    /// `by_node_handles_non_uniform_cut_projection` below are the two gates that
    /// compare a genuinely executed opening-block run to the trial-point path.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn by_node_degenerates_on_single_opening() {
        let case_dir = single_opening_case_dir();
        let stub = StubComm;

        let lb_by_node = train_final_lb(
            fresh_setup(case_dir, BackwardScheduler::ByNode { block_size: None }),
            1,
            &stub,
        );
        let lb_by_scenario = train_final_lb(
            fresh_setup(case_dir, BackwardScheduler::ByScenario {}),
            1,
            &stub,
        );

        assert_eq!(
            lb_by_node.to_bits(),
            lb_by_scenario.to_bits(),
            "by_node must degenerate to by_scenario bit-for-bit on a single-opening case"
        );
    }

    fn non_uniform_cut_projection_case_dir() -> &'static Path {
        Path::new("../../examples/deterministic/d43-storage-only-cut")
    }

    /// Non-uniform cut-projection gate: `d43-storage-only-cut` disables
    /// `inflow_lags` on one interior stage only, so successive backward
    /// stages solved by the SAME worker hand `process_stage_backward_by_node` (and
    /// `by_node_finish`) a `cut_n_state` that shrinks then regrows across
    /// stages — a stale-length buffer reuse across that change panics in
    /// `copy_from_slice`. `d43` is also single-opening per stage (like `d01` above),
    /// so `by_node` degenerates to a trial-point-equivalent unit here too, and
    /// `final_lb` must equal the `by_scenario` run bit-for-bit.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn by_node_handles_non_uniform_cut_projection() {
        let case_dir = non_uniform_cut_projection_case_dir();
        let stub = StubComm;

        let lb_by_node = train_final_lb(
            fresh_setup(case_dir, BackwardScheduler::ByNode { block_size: None }),
            1,
            &stub,
        );
        let lb_by_scenario = train_final_lb(
            fresh_setup(case_dir, BackwardScheduler::ByScenario {}),
            1,
            &stub,
        );

        assert_eq!(
            lb_by_node.to_bits(),
            lb_by_scenario.to_bits(),
            "by_node must handle a non-uniform per-stage cut projection and match \
             by_scenario bit-for-bit"
        );
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn by_node_falls_back_to_by_scenario_under_active_dcs() {
        let case_dir = fixture_case_dir();
        let stub = StubComm;

        let lb_by_node = train_final_lb(
            fresh_setup_with_active_dcs(case_dir, BackwardScheduler::ByNode { block_size: None }),
            1,
            &stub,
        );
        let lb_by_scenario = train_final_lb(
            fresh_setup_with_active_dcs(case_dir, BackwardScheduler::ByScenario {}),
            1,
            &stub,
        );

        assert_eq!(
            lb_by_node.to_bits(),
            lb_by_scenario.to_bits(),
            "by_node must degenerate to by_scenario bit-for-bit under active DCS"
        );
    }

    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn by_node_generates_one_cut_per_by_scenario() {
        let case_dir = fixture_case_dir();
        let stub = StubComm;

        let rows_by_node = train_rows_generated(
            fresh_setup_one_iteration(case_dir, BackwardScheduler::ByNode { block_size: None }),
            &stub,
        );
        let rows_by_scenario = train_rows_generated(
            fresh_setup_one_iteration(case_dir, BackwardScheduler::ByScenario {}),
            &stub,
        );

        assert_eq!(
            rows_by_node, rows_by_scenario,
            "by_node and by_scenario must generate the same number of cuts"
        );
    }

    /// The opening-block path's per-worker `backward_wall_ms` is observable through the
    /// event channel on the public `StudySetup::train` entry point (unlike
    /// `by_node_scratch`'s scratch-sizing gate below, which has no such
    /// surface), so this test drives the real training path rather than
    /// `BackwardPassState` directly.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn by_node_populates_backward_wall_ms() {
        let case_dir = fixture_case_dir();
        let stub = StubComm;
        let mut setup =
            fresh_setup_one_iteration(case_dir, BackwardScheduler::ByNode { block_size: None });
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let (event_tx, event_rx) = mpsc::channel::<TrainingEvent>();
        let outcome = setup
            .train(
                &mut solver,
                &stub,
                1,
                ActiveSolver::new,
                Some(event_tx),
                None,
            )
            .expect("train must return Ok");
        assert!(
            outcome.error.is_none(),
            "expected no training error, got: {:?}",
            outcome.error
        );

        let backward_walls: Vec<f64> = event_rx
            .try_iter()
            .filter_map(|e| match e {
                TrainingEvent::WorkerTiming {
                    phase: WorkerTimingPhase::Backward,
                    timings,
                    ..
                } => Some(timings.backward_wall_ms),
                _ => None,
            })
            .collect();

        assert!(
            !backward_walls.is_empty(),
            "expected at least one Backward WorkerTiming event"
        );
        assert!(
            backward_walls.iter().any(|&ms| ms > 0.0),
            "opening-block backward pass must populate backward_wall_ms > 0.0 for at least one worker, \
             got {backward_walls:?}"
        );
    }
}

mod by_node_scratch {
    //! `ByNodeScratch` sizing/gating/no-alloc gate. Unlike
    //! `by_node_scheduler_determinism`'s `examples/1dtoy`-based tests, these drive
    //! `BackwardPassState` directly against a small synthetic 2-stage, 2-opening
    //! fixture: `set_scheduler` and the opening-block scratch it sizes are internal to the
    //! backward pass and have no observable surface through the public
    //! `StudySetup::train` entry point.

    use std::num::NonZeroUsize;

    use cobre_core::scenario::{
        CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        SamplingScheme,
    };
    use cobre_core::temporal::{NoiseMethod, ScenarioSourceConfig};
    use cobre_core::{EntityId, HydroGenerationModel, SystemBuilder};
    use cobre_io::config::BackwardScheduler;
    use cobre_sddp::{
        BackwardPassInputs, BackwardPassState, ExchangeBuffers,
        context::{StageContext, TrainingContext},
        cut::FutureCostFunction,
        cut_sync::CutSyncBuffers,
        horizon_mode::HorizonMode,
        inflow_method::InflowNonNegativityMethod,
        risk_measure::RiskMeasure,
        test_support::{
            all_enabled_cut_state_layouts, state_layout, study_dims, trial_state_records,
        },
        workspace::{BasisStore, WorkspacePool, WorkspaceSizing},
    };
    use cobre_solver::{
        Basis, LpSolution, RowBatch, SolutionView, SolverError, SolverInterface, SolverStatistics,
        StageTemplate,
    };
    use cobre_stochastic::{ClassSchemes, OpeningTreeInputs, build_stochastic_context};

    use crate::common::StubComm;
    use crate::common::builders::{BusSpec, HydroSpec, make_bus, make_hydro, make_stage};

    /// Minimal `SolverInterface` mock returning a fixed feasible solution for
    /// every solve; mirrors `backward_pass_state.rs`'s own unit-test mock, not
    /// reachable from this external test crate. Tracks a per-solve
    /// `total_iterations` increment so `SolverStatsDelta::simplex_iterations`
    /// (and the block-pivot accumulator built on it) is genuinely non-zero.
    struct MockSolver {
        solution: LpSolution,
        current_num_rows: usize,
        buf_primal: Vec<f64>,
        buf_dual: Vec<f64>,
        buf_reduced_costs: Vec<f64>,
        total_iterations: u64,
    }

    impl MockSolver {
        fn always_ok(solution: LpSolution) -> Self {
            let base_rows = solution.dual.len();
            let buf_primal = solution.primal.clone();
            let buf_dual = solution.dual.clone();
            let buf_reduced_costs = solution.reduced_costs.clone();
            Self {
                solution,
                current_num_rows: base_rows,
                buf_primal,
                buf_dual,
                buf_reduced_costs,
                total_iterations: 0,
            }
        }
    }

    impl SolverInterface for MockSolver {
        type Profile = cobre_solver::ActiveProfile;

        fn apply_profile(&mut self, _profile: &cobre_solver::ActiveProfile) {}

        fn name(&self) -> &'static str {
            "mock"
        }
        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, template: &StageTemplate) {
            self.current_num_rows = template.num_rows;
            self.buf_primal = self.solution.primal.clone();
            self.buf_dual = self.solution.dual.clone();
            self.buf_reduced_costs = self.solution.reduced_costs.clone();
            self.buf_dual.resize(self.current_num_rows, 0.0);
        }
        fn add_rows(&mut self, cuts: &RowBatch) {
            self.current_num_rows += cuts.num_rows;
            self.buf_dual.resize(self.current_num_rows, 0.0);
        }
        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn solve(&mut self, _basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
            self.total_iterations += 3;
            Ok(SolutionView {
                objective: self.solution.objective,
                primal: &self.buf_primal,
                dual: &self.buf_dual,
                reduced_costs: &self.buf_reduced_costs,
                iterations: 0,
                solve_time_seconds: 0.0,
            })
        }
        fn get_basis(&mut self, out: &mut Basis) {
            *out = Basis::new(0, 0);
        }
        fn statistics(&self) -> SolverStatistics {
            SolverStatistics {
                total_iterations: self.total_iterations,
                ..SolverStatistics::default()
            }
        }
        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&self.statistics());
        }
    }

    /// Single-state-column template: one storage-like state column, one aux
    /// column pinned `[0, 0]` by an equality row, one zero-cost objective column
    /// — a trivially solvable LP any `SolverInterface` accepts unconditionally.
    fn minimal_template_1_0() -> StageTemplate {
        StageTemplate {
            num_cols: 3,
            num_rows: 1,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY; 3],
            objective: vec![0.0, 0.0, 1.0],
            row_lower: vec![0.0],
            row_upper: vec![0.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    fn solution_1_0(objective: f64, dual_storage: f64) -> LpSolution {
        LpSolution {
            objective,
            primal: vec![0.0, 0.0, 0.0],
            dual: vec![dual_storage],
            reduced_costs: vec![0.0; 3],
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    fn empty_cut_batches(n_stages: usize) -> Vec<RowBatch> {
        (0..n_stages)
            .map(|_| RowBatch {
                num_rows: 0,
                row_starts: Vec::new(),
                col_indices: Vec::new(),
                values: Vec::new(),
                row_lower: Vec::new(),
                row_upper: Vec::new(),
            })
            .collect()
    }

    /// Single-hydro, single-bus `StochasticContext` with `n_stages` monthly
    /// stages and `branching_factor` openings at every successor stage.
    fn make_stochastic_context(
        n_stages: usize,
        branching_factor: usize,
    ) -> cobre_stochastic::StochasticContext {
        use std::collections::BTreeMap;

        let bus = make_bus(EntityId(0), BusSpec::default());
        let hydro = make_hydro(
            EntityId(1),
            HydroSpec {
                bus_id: EntityId(0),
                max_storage_hm3: 100.0,
                max_turbined_m3s: 100.0,
                max_generation_mw: 100.0,
                generation_model: HydroGenerationModel::ConstantProductivity,
                ..HydroSpec::default()
            },
        );

        let stages: Vec<_> = (0..n_stages)
            .map(|idx| {
                make_stage(
                    idx,
                    crate::common::builders::StageSpec {
                        scenario_config: ScenarioSourceConfig {
                            branching_factor,
                            noise_method: NoiseMethod::Saa,
                        },
                        ..crate::common::builders::StageSpec::default()
                    },
                )
            })
            .collect();

        let inflow_models: Vec<_> = (0..n_stages)
            .map(|idx| InflowModel {
                hydro_id: EntityId(1),
                stage_id: idx as i32,
                mean_m3s: 100.0,
                std_m3s: 30.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            })
            .collect();

        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: vec![CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(1),
                    }],
                    matrix: vec![vec![1.0]],
                }],
            },
        );
        let correlation = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };

        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(correlation)
            .build()
            .expect("system must build");

        build_stochastic_context(
            &system,
            42,
            None,
            &[],
            &[],
            OpeningTreeInputs::default(),
            ClassSchemes {
                inflow: Some(SamplingScheme::InSample),
                load: Some(SamplingScheme::InSample),
                ncs: Some(SamplingScheme::InSample),
            },
        )
        .expect("stochastic context must build")
    }

    #[test]
    fn by_node_scratch_empty_on_by_scenario_default() {
        let mut state = BackwardPassState::new(1, 1, 4, 0, 3, 5, 2);
        state.set_scheduler(BackwardScheduler::ByScenario {});
        assert_eq!(
            state.by_node_scratch_arena_capacity(),
            0,
            "the default by_scenario scheduler must keep the opening-block scratch arena empty"
        );
        assert!(state.by_node_scratch_arena().is_empty());
        assert!(state.block_pivot_means().is_empty());
    }

    #[test]
    fn by_node_scratch_sized_from_study_dims() {
        let max_local_fwd = 3_usize;
        let bwd_max_openings = 4_usize;
        let n_state = 5_usize;
        let num_stages = 6_usize;
        let mut state = BackwardPassState::new(
            1,
            1,
            bwd_max_openings,
            0,
            max_local_fwd,
            n_state,
            num_stages,
        );

        state.set_scheduler(BackwardScheduler::ByNode { block_size: None });

        let arena = state.by_node_scratch_arena();
        assert_eq!(
            arena.len(),
            max_local_fwd * bwd_max_openings,
            "arena must hold max_local_fwd * bwd_max_openings outcomes"
        );
        let pivots = state.block_pivot_means();
        assert_eq!(
            pivots.len(),
            num_stages,
            "pivot means must have one row per stage"
        );
        assert!(
            pivots.iter().all(|row| row.len() == bwd_max_openings),
            "every stage row must be bwd_max_openings wide"
        );
        assert!(
            pivots.iter().flatten().all(Option::is_none),
            "a freshly sized accumulator must hold no populated entries"
        );
        assert!(
            arena
                .iter()
                .all(|outcome| outcome.coefficients.len() == n_state),
            "every outcome's coefficients must be pre-sized to n_state"
        );
    }

    /// Drives `BackwardPassState` directly across several repeated backward-pass
    /// runs under `ByNode`: the arena's `.capacity()` right after
    /// `set_scheduler` must equal its capacity after every run — no hot-path
    /// reallocation.
    #[test]
    #[cfg_attr(
        not(feature = "slow-tests"),
        ignore = "slow: run with --features slow-tests"
    )]
    fn by_node_scratch_capacity_stable_across_training() {
        let n_stages = 2_usize;
        let n_openings = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let state_layout_fixture = state_layout(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let frozen_templates = templates.clone();
        let base_rows = vec![1_usize; n_stages];
        let n_state = state_layout_fixture.n_state;
        let forward_passes = 2_u32;

        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
        let trial_states = vec![vec![10.0], vec![20.0]];
        let records = trial_state_records(&trial_states, n_stages);
        let mut exchange = ExchangeBuffers::new(n_state, trial_states.len(), 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let comm = StubComm;
        let mut workspace_pool = WorkspacePool::new(
            0,
            1,
            n_state,
            WorkspaceSizing {
                hydro_count: 1,
                max_openings: n_openings,
                initial_pool_capacity: 20,
                n_state,
                ..WorkspaceSizing::default()
            },
            || MockSolver::always_ok(solution_1_0(100.0, -5.0)),
        );
        let mut basis_store = BasisStore::new(exchange.local_count(), n_stages);
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let mut cut_batches = empty_cut_batches(n_stages);
        let ctx = StageContext {
            geometry_per_stage: &[],
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            cost_scale_factor: 1_000_000.0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            n_ncs: 0,
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            anticipated_windows: &[],
            study_stage_ids: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let study_dims_fixture = study_dims();
        let training_ctx = TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state_layout_fixture,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state_layout_fixture, n_stages),
            study_dims: &study_dims_fixture,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let local_count = exchange.local_count();
        let mut state =
            BackwardPassState::new(1, 1, n_openings, n_state, local_count, n_state, n_stages);
        state.set_scheduler(BackwardScheduler::ByNode {
            block_size: NonZeroUsize::new(1),
        });
        let capacity_after_set_scheduler = state.by_node_scratch_arena_capacity();
        assert!(
            capacity_after_set_scheduler > 0,
            "ByNode must size the opening-block scratch arena at set_scheduler time"
        );

        let mut inputs = BackwardPassInputs {
            workspaces: &mut workspace_pool.workspaces,
            basis_store: &mut basis_store,
            ctx: &ctx,
            frozen: &frozen_templates,
            fcf: &mut fcf,
            cut_batches: &mut cut_batches,
            training_ctx: &training_ctx,
            comm: &comm,
            exchange: &mut exchange,
            records: &records,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            event_sender: None,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            iteration: 1,
            local_work: local_count,
            fwd_offset: 0,
        };

        for iteration in 1..=3_u64 {
            inputs.iteration = iteration;
            let _ = state
                .run(&mut inputs)
                .expect("backward pass must not error");
            assert_eq!(
                state.by_node_scratch_arena_capacity(),
                capacity_after_set_scheduler,
                "opening-block scratch arena must not reallocate across repeated backward-pass runs \
                 (iteration {iteration})"
            );
        }
    }

    /// Build a fresh 2-stage, `n_openings`-opening direct-drive fixture under
    /// `ByNode` (block size 1, so every opening is its own block), run
    /// exactly one backward-pass iteration, and return the resulting
    /// [`BackwardPassState`] for the caller to inspect (e.g. via
    /// `block_pivot_means` or `by_node_scratch_arena_capacity`). `pub(crate)`:
    /// also reused by `by_node_scheduler_determinism`'s capacity-invariance check.
    pub(crate) fn run_by_node_one_iteration(n_openings: usize) -> BackwardPassState {
        let n_stages = 2_usize;
        let stochastic = make_stochastic_context(n_stages, n_openings);
        let state_layout_fixture = state_layout(1, 0);
        let templates = vec![minimal_template_1_0(); n_stages];
        let frozen_templates = templates.clone();
        let base_rows = vec![1_usize; n_stages];
        let n_state = state_layout_fixture.n_state;
        let forward_passes = 2_u32;

        let mut fcf =
            FutureCostFunction::new(n_stages, n_state, forward_passes, 20, &vec![0; n_stages]);
        let trial_states = vec![vec![10.0], vec![20.0]];
        let records = trial_state_records(&trial_states, n_stages);
        let mut exchange = ExchangeBuffers::new(n_state, trial_states.len(), 1);
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let risk_measures = vec![RiskMeasure::Expectation; n_stages];

        let comm = StubComm;
        let mut workspace_pool = WorkspacePool::new(
            0,
            1,
            n_state,
            WorkspaceSizing {
                hydro_count: 1,
                max_openings: n_openings,
                initial_pool_capacity: 20,
                n_state,
                ..WorkspaceSizing::default()
            },
            || MockSolver::always_ok(solution_1_0(100.0, -5.0)),
        );
        let mut basis_store = BasisStore::new(exchange.local_count(), n_stages);
        let mut csb = CutSyncBuffers::with_distribution(n_state, 64, 1, exchange.local_count());
        let mut cut_batches = empty_cut_batches(n_stages);
        let ctx = StageContext {
            geometry_per_stage: &[],
            templates: &templates,
            base_rows: &base_rows,
            noise_scale: &[],
            n_hydros: 0,
            cost_scale_factor: 1_000_000.0,
            n_load_buses: 0,
            load_balance_row_starts: &[],
            load_bus_indices: &[],
            block_counts_per_stage: &[],
            ncs_col_starts: &[],
            n_ncs: 0,
            ncs_stochastic_dense_col: &[],
            ncs_stochastic_windows: &[],
            anticipated_windows: &[],
            study_stage_ids: &[],
            ncs_max_gen: &[],
            ncs_allow_curtailment: &[],
            discount_factors: &[],
            cumulative_discount_factors: &[],
            stage_lag_transitions: &[],
            noise_group_ids: &[],
            downstream_par_order: 0,
        };
        let study_dims_fixture = study_dims();
        let training_ctx = TrainingContext {
            node_graph: &cobre_sddp::test_support::chain_node_graph(&stochastic),
            horizon: &horizon,
            state: &state_layout_fixture,
            cut_state_layouts: &all_enabled_cut_state_layouts(&state_layout_fixture, n_stages),
            study_dims: &study_dims_fixture,
            inflow_method: &InflowNonNegativityMethod::None,
            stochastic: &stochastic,
            initial_state: &[],
            inflow_scheme: SamplingScheme::InSample,
            load_scheme: SamplingScheme::InSample,
            ncs_scheme: SamplingScheme::InSample,
            stages: &[],
            historical_library: None,
            external_inflow_library: None,
            external_load_library: None,
            external_ncs_library: None,
            lag_accum_seed: &[],
            lag_weight_seed: &[],
            dcs: None,
        };

        let local_count = exchange.local_count();
        let mut state =
            BackwardPassState::new(1, 1, n_openings, n_state, local_count, n_state, n_stages);
        state.set_scheduler(BackwardScheduler::ByNode {
            block_size: NonZeroUsize::new(1),
        });

        let mut inputs = BackwardPassInputs {
            workspaces: &mut workspace_pool.workspaces,
            basis_store: &mut basis_store,
            ctx: &ctx,
            frozen: &frozen_templates,
            fcf: &mut fcf,
            cut_batches: &mut cut_batches,
            training_ctx: &training_ctx,
            comm: &comm,
            exchange: &mut exchange,
            records: &records,
            cut_sync_bufs: &mut csb,
            visited_archive: None,
            event_sender: None,
            risk_measures: &risk_measures,
            cut_activity_tolerance: 0.0,
            iteration: 1,
            local_work: local_count,
            fwd_offset: 0,
        };

        let _ = state
            .run(&mut inputs)
            .expect("backward pass must not error");
        state
    }

    /// `BackwardPassState`'s block-pivot accumulator has no surface through the
    /// public `StudySetup::train` entry point (`block_pivot_means` is a
    /// `BackwardPassState` accessor, and `BackwardPassState` is internal to
    /// `TrainingSession` — see this module's doc comment), so this test drives
    /// it directly, mirroring `by_node_scratch_capacity_stable_across_training`
    /// above, rather than training `examples/1dtoy` via the public API.
    ///
    /// Two independently-constructed fixtures, each trained for one iteration,
    /// must produce bit-identical `(stage, block)` mean-pivot accumulators
    /// (integer sum/count accumulation is exactly reproducible), and the
    /// accumulator must hold at least one populated (non-`None`) entry.
    #[test]
    fn block_pivots_reproducible_and_populated() {
        let n_openings = 4_usize;

        let means_a = run_by_node_one_iteration(n_openings).block_pivot_means();
        let means_b = run_by_node_one_iteration(n_openings).block_pivot_means();

        assert_eq!(
            means_a, means_b,
            "two independent runs of the same fixture must produce bit-identical pivot \
             accumulators"
        );
        assert!(
            means_a.iter().flatten().any(Option::is_some),
            "the pivot accumulator must hold at least one populated (stage, block) entry, \
             got {means_a:?}"
        );
    }
}

#[cfg(feature = "test-support")]
mod k_fan_graph_invariance {
    //! Thread-shape invariance on the DECOMP K-fan graph path (sampled): the
    //! end-to-end complement to the `n_workers ∈ {1,2}` per-pool-routing unit
    //! test, certifying the per-pool trial-state routing and the
    //! reverse-topological backward level's own batching are
    //! worker-count-invariant on a genuinely branching graph. Also pins the
    //! corpus-unstated obligation that the LB root evaluation, the
    //! outcome-allgather aggregation, and the weighted forward sync each
    //! reduce in canonical order — invariant across several distinct
    //! worker-partition boundaries, not merely a single threads=1-vs-k pair.

    use cobre_sddp::test_support::k_fan_setup;
    use cobre_solver::ActiveSolver;

    use crate::common::StubComm;

    const K: usize = 8;
    const FORWARD_PASSES: u32 = 6;
    const MAX_ITERATIONS: u32 = 3;

    /// Train a fresh K-fan fixture at `n_threads`, returning
    /// `(final_lb, final_ub, final_ub_std)` — the LB root evaluation and the
    /// statistical (Welford) upper bound's mean/std, both fed by the
    /// outcome-allgather aggregation and the weighted forward sync.
    fn run_shape(n_threads: usize) -> (f64, f64, f64) {
        let mut fixture = k_fan_setup(K, FORWARD_PASSES, MAX_ITERATIONS);
        let comm = StubComm;
        let mut solver = ActiveSolver::new().expect("ActiveSolver::new must succeed");
        let outcome = fixture
            .setup
            .train(&mut solver, &comm, n_threads, ActiveSolver::new, None, None)
            .expect("k_fan training must return Ok");
        assert!(
            outcome.error.is_none(),
            "k_fan training must not error: {:?}",
            outcome.error
        );
        (
            outcome.result.final_lb,
            outcome.result.final_ub,
            outcome.result.final_ub_std,
        )
    }

    /// Power precondition shared by both gates below: the fan stage carries
    /// `K` cut-generating nodes beyond the root (`>= 2`), and `forward_passes`
    /// exceeds the largest thread count crossed, so a multi-worker shape
    /// genuinely splits trial points across worker-partition boundaries
    /// instead of degenerating to one trial point per worker.
    fn assert_genuine_multi_node_level(max_threads_crossed: usize) {
        let probe = k_fan_setup(K, FORWARD_PASSES, MAX_ITERATIONS);
        let node_graph = &probe.setup.node_graph;
        let cut_generating = (0..node_graph.nodes.len())
            .filter(|&pos| !node_graph.successors[pos].is_empty())
            .count();
        let fan_nodes = cut_generating - 1;
        assert!(
            fan_nodes >= 2,
            "power precondition: the K-fan must carry >= 2 cut-generating fan nodes \
             (got {fan_nodes}) for thread-shape crossing to have power"
        );
        assert!(
            usize::try_from(FORWARD_PASSES).expect("FORWARD_PASSES fits usize")
                > max_threads_crossed,
            "power precondition: forward_passes ({FORWARD_PASSES}) must exceed the largest \
             thread count crossed ({max_threads_crossed}), or every worker gets at most one \
             trial point and no partition boundary is genuinely crossed"
        );
    }

    /// Thread-shape invariance: `--threads 1` vs `--threads k`, plus a
    /// same-shape repeat, bitwise-identical `final_lb`/`final_ub`/`final_ub_std`
    /// on the DECOMP K-fan graph path.
    #[test]
    #[ignore = "TODO(k-fan-per-node-frozen-templates): the forward_pass_index compaction (the \
                double-insert) is fixed, but a separate node-native gap remains — \
                freeze_active_cuts_into_templates (training/session/mod.rs) freezes ONE pool per \
                STAGE via node_graph.nodes[t].pool_id, using the stage index t as a node \
                POSITION. On a chain position==stage so it is correct; on a branching graph a \
                stage has several nodes with distinct pools, so the leaf stage's frozen template \
                embeds a fan node's cuts (nodes[2] is fan pool 2 in k_fan_setup(8,6,3)), and once \
                that pool holds a cut with intercept>0 the terminal leaf's pinned theta makes the \
                LP infeasible at iteration 3. Needs per-node/per-pool frozen templates + per-node \
                forward LP load; out of scope for the forward_pass_index fix."]
    fn k_fan_thread_shape_invariance() {
        assert_genuine_multi_node_level(4);

        let (lb_k, ub_k, ub_std_k) = run_shape(4);
        let (lb_1, ub_1, ub_std_1) = run_shape(1);
        let (lb_repeat, ub_repeat, ub_std_repeat) = run_shape(1);

        assert_eq!(
            lb_k.to_bits(),
            lb_1.to_bits(),
            "final_lb must be bitwise identical across thread shapes"
        );
        assert_eq!(
            ub_k.to_bits(),
            ub_1.to_bits(),
            "final_ub must be bitwise identical across thread shapes"
        );
        assert_eq!(
            ub_std_k.to_bits(),
            ub_std_1.to_bits(),
            "final_ub_std must be bitwise identical across thread shapes"
        );
        assert_eq!(
            lb_1.to_bits(),
            lb_repeat.to_bits(),
            "same-shape repeat final_lb must be bitwise identical"
        );
        assert_eq!(
            ub_1.to_bits(),
            ub_repeat.to_bits(),
            "same-shape repeat final_ub must be bitwise identical"
        );
        assert_eq!(
            ub_std_1.to_bits(),
            ub_std_repeat.to_bits(),
            "same-shape repeat final_ub_std must be bitwise identical"
        );
    }

    /// Weighted-aggregation canonical-order obligation: the LB root
    /// evaluation, the outcome-allgather aggregation, and the weighted
    /// forward sync each reduce in canonical order on the graph path, so
    /// `final_lb`/`final_ub`/`final_ub_std` stay bitwise identical across
    /// FOUR distinct thread-partition boundaries (1, 2, 3, 4) — not merely
    /// the single threads=1-vs-k pair the sibling gate above crosses.
    #[test]
    #[ignore = "TODO(k-fan-per-node-frozen-templates): blocked on the same per-stage frozen \
                template gap as k_fan_thread_shape_invariance (freeze_active_cuts_into_templates \
                keys pools by stage-as-node-position); training the K-fan reaches an infeasible \
                iteration-3 leaf before this assertion runs. The forward_pass_index compaction is \
                fixed; this needs per-node frozen templates."]
    fn k_fan_weighted_aggregation_canonical_order_invariance() {
        assert_genuine_multi_node_level(4);

        let shapes = [1usize, 2, 3, 4];
        let results: Vec<(f64, f64, f64)> = shapes.iter().copied().map(run_shape).collect();

        let (lb0, ub0, ub_std0) = results[0];
        for (&n_threads, &(lb, ub, ub_std)) in shapes.iter().zip(&results).skip(1) {
            assert_eq!(
                lb.to_bits(),
                lb0.to_bits(),
                "final_lb (the LB root evaluation) must be bitwise identical between \
                 threads=1 and threads={n_threads}"
            );
            assert_eq!(
                ub.to_bits(),
                ub0.to_bits(),
                "final_ub (the weighted forward sync / outcome-allgather aggregation) must \
                 be bitwise identical between threads=1 and threads={n_threads}"
            );
            assert_eq!(
                ub_std.to_bits(),
                ub_std0.to_bits(),
                "final_ub_std must be bitwise identical between threads=1 and \
                 threads={n_threads}"
            );
        }
    }
}
