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

        let stage0_unpacked = results[0][0].as_ref().expect("rank 0 stage 0 must be Some");
        assert_captured_basis_eq(&stage0_basis, stage0_unpacked, "pack/unpack parity stage 0");

        let stage2_unpacked = results[0][2].as_ref().expect("rank 0 stage 2 must be Some");
        assert_captured_basis_eq(&stage2_basis, stage2_unpacked, "pack/unpack parity stage 2");
    }
}

#[cfg(all(feature = "highs", feature = "test-support"))]
mod retry_armed_determinism {
    //! Retry-armed determinism gate: the `backward_tuned_v1` preset with a
    //! deliberately low `simplex_iteration_limit` forces the `HiGHS` retry
    //! escalation ladder (`crates/cobre-solver/src/backends/highs/retry.rs`)
    //! to fire, then asserts the final training lower bound is bitwise
    //! identical across four execution shapes of the SAME config: threads=k,
    //! threads=1, a same-shape repeat, and a faithful 2-rank leg. Runs on both
    //! an expectation and a `CVaR` risk configuration of the same fixture.
    //!
    //! Power statement: this gate catches a profile option dropped at the
    //! retry-finalization seam (`reapply_profile` after
    //! `restore_default_settings`) that manifests only once a solve genuinely
    //! retries; it has no power on a case/length that produces zero retries.
    //! `total_retries > 0` is asserted per shape as the gate's own
    //! self-check, not an incidental fact.

    use std::path::Path;

    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_core::scenario::ScenarioSource;
    use cobre_io::config::PhaseSolverProfileConfig;
    use cobre_sddp::{
        Phase, RiskMeasure, SolverProfiles, StudySetup, hydro_models::prepare_hydro_models,
        setup::prepare_stochastic,
    };
    use cobre_solver::ActiveSolver;

    use crate::common::{StubComm, build_setup_for_case};

    /// Low enough that the tuned `backward_tuned_v1` profile's first attempt
    /// cannot finish within the cap on every stage solve of the d03 fixture,
    /// arming the retry-escalation ladder on every solve rather than
    /// occasionally; tuned empirically against this fixture, not derived
    /// from a closed form.
    const FORCED_SIMPLEX_ITERATION_LIMIT: u32 = 1;

    /// `backward_tuned_v1` (`SteepestEdge` / Curtis-Reid / Row / ptol `1e-7`)
    /// forced to [`FORCED_SIMPLEX_ITERATION_LIMIT`]. `forward` stays
    /// byte-neutral (`Phase::Forward.resolve_profile(None)`): only the
    /// backward-pass retry-finalization seam is under test.
    fn forced_retry_profiles() -> SolverProfiles {
        let tuned = PhaseSolverProfileConfig {
            preset: Some("backward_tuned_v1".to_string()),
            dual_edge_weight: None,
            scale: None,
            price: None,
            primal_feasibility_tolerance: None,
        };
        let mut backward = Phase::Backward.resolve_profile(Some(&tuned));
        backward.simplex_iteration_limit = FORCED_SIMPLEX_ITERATION_LIMIT;
        SolverProfiles {
            forward: Phase::Forward.resolve_profile(None),
            backward,
        }
    }

    fn d03_case_dir() -> &'static Path {
        Path::new("../../examples/deterministic/d03-two-hydro-cascade")
    }

    /// Build a fresh [`StudySetup`], mirroring `deterministic.rs`'s
    /// `run_deterministic_with_solver` construction pipeline. Each shape
    /// trains from an independently-built setup, never a warm-started reuse,
    /// so the four shapes compare cold-to-cold.
    fn fresh_setup(case_dir: &Path) -> StudySetup {
        let config_path = case_dir.join("config.json");
        let config = cobre_io::parse_config(&config_path).expect("config must parse");
        let system = cobre_io::load_case(case_dir).expect("load_case must succeed");

        let prepare_result =
            prepare_stochastic(system, case_dir, &config, 42, &ScenarioSource::default())
                .expect("prepare_stochastic must succeed");
        let system = prepare_result.system;
        let stochastic = prepare_result.stochastic;

        let hydro_models = prepare_hydro_models(&system, case_dir, false)
            .expect("prepare_hydro_models must succeed");

        build_setup_for_case(case_dir, &config, &system, stochastic, hydro_models)
    }

    /// `size() == 2` sibling of [`StubComm`]: every collective writes only
    /// this rank's own slot (`recv[displs[0]..displs[0] + send.len()]`),
    /// mirroring `state_exchange.rs`'s own `Rank1Of2` test pattern rather than
    /// echoing rank 0's data into rank 1's slot. Faithful — not a dishonest
    /// tautology — only because every fixture below trains with
    /// `forward_passes == 1`: `RankDistribution` (`base_fwd=0, remainder=1`
    /// for `num_ranks=2`) assigns rank 0 the sole real forward pass and rank 1
    /// exactly zero, so the zero contribution this stub leaves unwritten IS
    /// what a genuine rank 1 would also send.
    struct Rank0Of2;

    impl Communicator for Rank0Of2 {
        fn allgatherv<T: CommData>(
            &self,
            send: &[T],
            recv: &mut [T],
            _counts: &[usize],
            displs: &[usize],
        ) -> Result<(), CommError> {
            let start = displs[0];
            recv[start..start + send.len()].clone_from_slice(send);
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
            2
        }

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
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
    /// `cvar_alpha_half_concentrates_on_worst` fixture — first `CVaR`
    /// determinism coverage in the suite.
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
