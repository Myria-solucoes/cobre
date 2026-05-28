//! Integration smoke for the hybrid basis-reconstruction path.
//!
//! Verifies that [`cobre_sddp::basis_reconstruct::reconstruct_basis_hybrid`]
//! produces the expected statistics and output statuses on a mixed
//! preserved-plus-new-slot fixture.  The test exercises the public hybrid
//! function directly without invoking the full SDDP training loop, so it
//! runs in milliseconds and remains stable across solver versions.
//!
//! ## Why a public-API integration test
//!
//! The hybrid path is selected at the `stage_solve` call site at compile
//! time.  Because Cargo cannot link the same binary with two different
//! feature flags, the cross-feature comparison (legacy vs hybrid) happens
//! externally — this test file checks only the hybrid behaviour, gated
//! behind `#[cfg(feature = "basis-hybrid")]`.
//!
//! The default `cargo test` run skips this file entirely (no tests are
//! defined outside the feature gate).  Pass `--features basis-hybrid` to
//! include the suite.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation
)]

#[cfg(feature = "basis-hybrid")]
mod hybrid {
    use cobre_sddp::basis_reconstruct::{
        HIGHS_BASIS_STATUS_BASIC as B, HIGHS_BASIS_STATUS_LOWER as L, ReconstructionStats,
        ReconstructionTarget, reconstruct_basis_hybrid,
    };
    use cobre_sddp::workspace::CapturedBasis;
    use cobre_solver::Basis;

    /// Build a `CapturedBasis` populated with the requested slot list and
    /// cut-row status sequence.  Template rows are all `BASIC`; columns are
    /// all `BASIC` so the column block does not perturb the test focus.
    fn make_stored(
        base_rows: usize,
        num_cols: usize,
        slots: &[u32],
        cut_statuses: &[i32],
        state_at_capture: &[f64],
    ) -> CapturedBasis {
        assert_eq!(slots.len(), cut_statuses.len());
        let total_rows = base_rows + cut_statuses.len();
        let mut cb = CapturedBasis::new(
            num_cols,
            total_rows,
            base_rows,
            slots.len(),
            state_at_capture.len(),
        );
        cb.basis.row_status.clear();
        cb.basis.row_status.resize(base_rows, B);
        cb.basis.row_status.extend_from_slice(cut_statuses);
        cb.basis.col_status.clear();
        cb.basis.col_status.resize(num_cols, B);
        cb.cut_row_slots.extend_from_slice(slots);
        cb.state_at_capture.extend_from_slice(state_at_capture);
        cb
    }

    /// 5-row mixed-case acceptance fixture: stored preserves slots
    /// `{10, 20, 30, 40}` with statuses `[L, B, L, B]`; the target LP has
    /// 5 cut rows for slots `{10, 25, 30, 45, 50}`.  The hybrid path must
    /// copy stored statuses for the preserved slots and assign `BASIC`
    /// to the three new slots.
    #[test]
    fn hybrid_mixed_case_5_rows() {
        let stored = make_stored(2, 3, &[10, 20, 30, 40], &[L, B, L, B], &[1.0, 2.0]);
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (10, 0.0, vec![0.0, 0.0]),
            (25, 0.0, vec![0.0, 0.0]),
            (30, 0.0, vec![0.0, 0.0]),
            (45, 0.0, vec![0.0, 0.0]),
            (50, 0.0, vec![0.0, 0.0]),
        ];
        let target = ReconstructionTarget {
            base_row_count: 2,
            num_cols: 3,
        };
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 64];

        let stats = reconstruct_basis_hybrid(
            &stored,
            target,
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            &mut out,
            &mut lookup,
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 2,
                new_tight: 0,
                new_slack: 3,
            },
            "preserved={{10, 30}} (2), new_slack={{25, 45, 50}} (3)",
        );
        assert_eq!(out.row_status.len(), 7, "2 template + 5 cut rows");
        // Template block first (both BASIC from the stored basis).
        assert_eq!(&out.row_status[..2], &[B, B], "template rows preserved");
        // Cut block: stored slot-10 = L, new slot 25 = B,
        // stored slot-30 = L, new slot 45 = B, new slot 50 = B.
        assert_eq!(out.row_status[2], L, "slot 10 → stored LOWER");
        assert_eq!(out.row_status[3], B, "slot 25 → new → BASIC");
        assert_eq!(out.row_status[4], L, "slot 30 → stored LOWER");
        assert_eq!(out.row_status[5], B, "slot 45 → new → BASIC");
        assert_eq!(out.row_status[6], B, "slot 50 → new → BASIC");
    }

    /// All slots preserved: every cut row in the target LP has a matching
    /// stored slot, so the output row statuses are an exact copy of the
    /// stored cut-row block and `new_slack == 0`.
    #[test]
    fn hybrid_all_preserved() {
        let stored = make_stored(1, 2, &[10, 20, 30], &[L, B, L], &[1.0]);
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (10, 0.0, vec![0.0]),
            (20, 0.0, vec![0.0]),
            (30, 0.0, vec![0.0]),
        ];
        let target = ReconstructionTarget {
            base_row_count: 1,
            num_cols: 2,
        };
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 64];

        let stats = reconstruct_basis_hybrid(
            &stored,
            target,
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            &mut out,
            &mut lookup,
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 3,
                new_tight: 0,
                new_slack: 0,
            },
        );
        assert_eq!(&out.row_status[1..], &[L, B, L]);
    }

    /// All slots new: stored is empty, every cut row gets `BASIC` and the
    /// `new_slack` counter equals the cut-row count.
    #[test]
    fn hybrid_all_new() {
        let stored = make_stored(1, 2, &[], &[], &[1.0]);
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (5, 0.0, vec![0.0]),
            (6, 0.0, vec![0.0]),
            (7, 0.0, vec![0.0]),
        ];
        let target = ReconstructionTarget {
            base_row_count: 1,
            num_cols: 2,
        };
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 64];

        let stats = reconstruct_basis_hybrid(
            &stored,
            target,
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            &mut out,
            &mut lookup,
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 0,
                new_tight: 0,
                new_slack: 3,
            },
        );
        assert_eq!(&out.row_status[1..], &[B, B, B]);
    }
}
