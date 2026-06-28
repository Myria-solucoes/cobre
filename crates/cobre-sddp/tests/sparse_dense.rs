//! `build_cut_row_batch_into` emits one `(lp_column, coefficient)` pair per
//! nonzero-state-mask entry in ascending order, then the theta column. For a
//! proper-subset mask (mixed AR orders), the emitted columns and values must
//! equal the remapped LP columns and the negated coefficients.

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
use cobre_sddp::build_cut_row_batch_into;
use cobre_sddp::indexer::StateLayout;
use cobre_solver::RowBatch;

#[test]
fn sparse_partial_mask_produces_correct_subset() {
    let n_hydro = 3;
    let max_par_order = 2;
    let n_state = n_hydro * (1 + max_par_order);

    // The [0, 1, 2] arg is per-hydro effective_lag_count; mixed orders zero out
    // some lag slots. StateLayout::new finalizes the mask as production setup does.
    let state = StateLayout::new(n_hydro, max_par_order, 0, 0, vec![], &[0, 1, 2]);
    // Expected mask = storage [0,1,2] + lag0 of h1,h2 [4,5] + lag1 of h2 [8].
    let mask = &state.nonzero_state_indices;
    assert_eq!(
        mask.len(),
        6,
        "mixed-order mask has 3 + 0 + 1 + 2 = 6 entries"
    );

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
    let cut_state = cobre_sddp::indexer::test_fixtures::cut_state_layout(&state);
    build_cut_row_batch_into(&mut batch, &fcf, 0, &state, &cut_state, &col_scale);

    assert_eq!(batch.num_rows, 1);
    let theta_col = state.theta;
    let expected_cols: Vec<i32> = mask
        .iter()
        .map(|&j| state.state_to_lp_column(j) as i32)
        .chain(std::iter::once(theta_col as i32))
        .collect();
    assert_eq!(
        batch.col_indices, expected_cols,
        "sparse col_indices mismatch"
    );

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
