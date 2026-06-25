//! Integration smoke for the slot-identity basis-reconstruction path.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation
)]

use cobre_sddp::basis_reconstruct::{
    HIGHS_BASIS_STATUS_BASIC as B, HIGHS_BASIS_STATUS_LOWER as L, ReconstructionStats,
    ReconstructionTarget, reconstruct_basis,
};
use cobre_sddp::workspace::CapturedBasis;
use cobre_solver::Basis;

/// All columns are `BASIC` so the column block does not perturb the test focus.
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

#[test]
fn mixed_case_5_rows() {
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

    let stats = reconstruct_basis(
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
    assert_eq!(&out.row_status[..2], &[B, B], "template rows preserved");
    assert_eq!(out.row_status[2], L, "slot 10 → stored LOWER");
    assert_eq!(out.row_status[3], B, "slot 25 → new → BASIC");
    assert_eq!(out.row_status[4], L, "slot 30 → stored LOWER");
    assert_eq!(out.row_status[5], B, "slot 45 → new → BASIC");
    assert_eq!(out.row_status[6], B, "slot 50 → new → BASIC");
}

#[test]
fn all_preserved() {
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

    let stats = reconstruct_basis(
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

#[test]
fn all_new() {
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

    let stats = reconstruct_basis(
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
