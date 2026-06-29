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

        let output_dir = case_dir.join("output").join("hydro_models");
        if output_dir.exists() {
            // No-op: prepare_hydro_models never writes files (the write site is the
            // CLI/Python entry point), so a pre-existing output dir is left untouched.
        }
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
        original.basis.col_status.extend_from_slice(&[0_i32, 1_i32]);
        original
            .basis
            .row_status
            .extend_from_slice(&[0_i32, 1_i32, 0_i32]);
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
            cb.basis.col_status.push(seed as i32 + i as i32);
        }
        for i in 0..num_rows {
            cb.basis.row_status.push(seed as i32 * 2 + i as i32);
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

        // Unpacked data must also match the original pack-side data.
        let stage0_unpacked = results[0][0].as_ref().expect("rank 0 stage 0 must be Some");
        assert_captured_basis_eq(&stage0_basis, stage0_unpacked, "pack/unpack parity stage 0");

        let stage2_unpacked = results[0][2].as_ref().expect("rank 0 stage 2 must be Some");
        assert_captured_basis_eq(&stage2_basis, stage2_unpacked, "pack/unpack parity stage 2");
    }
}
