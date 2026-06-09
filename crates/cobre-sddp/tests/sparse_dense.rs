//! Sparse cut-row correctness test.
//!
//! `build_cut_row_batch_into` emits one `(lp_column, coefficient)` pair per
//! entry of the indexer's nonzero-state mask (in ascending order), followed by
//! the theta column. This test verifies that, for a proper-subset mask (mixed
//! AR orders, so some lag slots are excluded), the emitted columns and values
//! match the remapped LP columns and negated coefficients exactly.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown
)]

use cobre_sddp::FutureCostFunction;
use cobre_sddp::StageIndexer;
use cobre_sddp::build_cut_row_batch_into;
use cobre_solver::RowBatch;

/// When the sparse mask is a proper subset (mixed AR orders), the output
/// contains fewer nonzeros per row than the full state dimension. This test
/// verifies the cut-row builder produces correct output for a mixed-order case.
#[test]
fn sparse_partial_mask_produces_correct_subset() {
    // 3 hydros, max_par_order = 2 → n_state = 3 * (1 + 2) = 9
    let n_hydro = 3;
    let max_par_order = 2;
    let n_state = n_hydro * (1 + max_par_order); // 9

    // Mixed AR orders: [0, 1, 2] → some lag slots are zero
    let mut indexer = StageIndexer::new(n_hydro, max_par_order);
    indexer.set_nonzero_mask(&[0, 1, 2], &[]);
    // Finalize the state→LP-column precompute map, as production setup does.
    indexer.finalize_state_column_map();
    // Expected mask: storage [0,1,2] + lag0 for h1,h2 [4,5] + lag1 for h2 [8]
    // = [0, 1, 2, 4, 5, 8]
    let mask = &indexer.nonzero_state_indices;
    assert_eq!(
        mask.len(),
        6,
        "mixed-order mask has 3 + 0 + 1 + 2 = 6 entries"
    );

    // Create FCF: new(num_stages, state_dim, fwd_passes, max_iter, warm_start).
    let mut fcf = FutureCostFunction::new(2, n_state, 1, 1, &[0; 2]);
    let coeffs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    fcf.add_cut(0, 0, 0, 50.0, &coeffs);

    let col_scale: Vec<f64> = Vec::new();

    let mut batch = RowBatch {
        num_rows: 0,
        row_starts: Vec::new(),
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };
    build_cut_row_batch_into(&mut batch, &fcf, 0, &indexer, &col_scale);

    // The sparse path should only emit entries for mask indices + theta.
    // NNZ per cut = mask.len() + 1 (theta) = 7
    assert_eq!(batch.num_rows, 1);
    // col_indices should contain the remapped LP columns plus theta_col.
    // state_to_lp_column maps outgoing-state indices to LP columns:
    // storage → identity; lag 0 → z_inflow; lag l≥1 → incoming lag l−1.
    let theta_col = indexer.theta;
    let expected_cols: Vec<i32> = mask
        .iter()
        .map(|&j| indexer.state_to_lp_column(j) as i32)
        .chain(std::iter::once(theta_col as i32))
        .collect();
    assert_eq!(
        batch.col_indices, expected_cols,
        "sparse col_indices mismatch"
    );

    // Values should be -coefficients[j] for each mask index, plus +1.0 for theta.
    let expected_values: Vec<f64> = mask
        .iter()
        .map(|&j| -coeffs[j])
        .chain(std::iter::once(1.0))
        .collect();
    for (i, (actual, expected)) in batch.values.iter().zip(expected_values.iter()).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "values[{i}] differ: actual={actual}, expected={expected}"
        );
    }
}
