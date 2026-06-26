//! Criterion micro-benchmark for `build_cut_row_batch_into` with the
//! anticipated-state block enabled.
//!
//! ## Design: `n_state`-matched comparison
//!
//! Both cases are constructed at **exactly the same `n_state` = 130** (and the
//! same `nnz_per_cut` = 131), so the timing ratio isolates per-coefficient
//! code-path overhead, not total state-vector size. The baseline reaches 130 via
//! N=10 hydros at full lag L=12 (dense path); the extended case via N=10 at L=2
//! plus `n_anticipated`=10 × `K_max`=10 anticipated slots (sparse path). Changing
//! either case's sizing breaks the comparison.
//!
//! ## Acceptance criterion
//!
//! The extended/baseline mean-time ratio must lie in `[0.5, 1.60]`. The
//! anticipated-state path carries ~46% per-coefficient overhead vs. the
//! storage/lag path (an optimization target, not a correctness issue); a ratio
//! above 1.60 is a new regression, below 0.5 a likely measurement artifact.

#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use cobre_sddp::build_cut_row_batch_into;
use cobre_sddp::cut::fcf::FutureCostFunction;
use cobre_sddp::indexer::StateLayout;
use cobre_solver::RowBatch;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

const N: usize = 10;
const L_BASELINE: usize = 12;
const L_ANTICIPATED: usize = 2;
const N_ANTICIPATED: usize = 10;
const K_MAX: usize = 10;
const N_STATE: usize = 130;

const NUM_CUTS: u32 = 50;

fn build_fcf(state_dimension: usize) -> FutureCostFunction {
    let mut fcf = FutureCostFunction::new(1, state_dimension, NUM_CUTS, 1, &[0]);
    let coefficients: Vec<f64> = (0..state_dimension).map(|j| 1.0 + j as f64).collect();
    for fp in 0..NUM_CUTS {
        fcf.add_cut(0, 0, fp, f64::from(fp), &coefficients);
    }
    fcf
}

fn build_row_batch(nnz_per_cut: usize) -> RowBatch {
    let n = NUM_CUTS as usize;
    let total_nnz = n * nnz_per_cut;
    RowBatch {
        num_rows: 0,
        row_starts: Vec::with_capacity(n + 1),
        col_indices: Vec::with_capacity(total_nnz),
        values: Vec::with_capacity(total_nnz),
        row_lower: Vec::with_capacity(n),
        row_upper: Vec::with_capacity(n),
    }
}

fn bench_cut_application_baseline(c: &mut Criterion) {
    // `StateLayout::new` finalizes the mask and column-map cache in its
    // constructor, mirroring production `build_wired_indexer`.
    let lag_counts: Vec<usize> = vec![L_BASELINE; N];
    let state = StateLayout::new(N, L_BASELINE, 0, 0, vec![], &lag_counts);
    debug_assert_eq!(
        state.n_state, N_STATE,
        "baseline n_state must equal {N_STATE}"
    );
    debug_assert_eq!(
        state.nonzero_state_indices.len(),
        N_STATE,
        "baseline mask must cover all {N_STATE} state dims"
    );

    let fcf = build_fcf(state.n_state);
    let nnz_per_cut = state.nonzero_state_indices.len() + 1;
    let mut batch = build_row_batch(nnz_per_cut);

    c.bench_function("bench_cut_application_baseline", |b| {
        b.iter(|| {
            build_cut_row_batch_into(
                &mut batch,
                black_box(&fcf),
                black_box(0),
                black_box(&state),
                black_box(&[]),
            );
            black_box(batch.values.len());
        });
    });
}

fn bench_cut_application_with_anticipated(c: &mut Criterion) {
    // All anticipated plants at K_i = K_max so every slot 0..K_max is nonzero;
    // this keeps the mask fully dense at n_state = 130, matching the baseline.
    let anticipated_lead_stages: Vec<usize> = vec![K_MAX; N_ANTICIPATED];

    // `StateLayout::new` finalizes both layout caches in its constructor.
    let lag_counts: Vec<usize> = vec![L_ANTICIPATED; N];
    let state = StateLayout::new(
        N,
        L_ANTICIPATED,
        N_ANTICIPATED,
        K_MAX,
        anticipated_lead_stages,
        &lag_counts,
    );
    debug_assert_eq!(
        state.n_state, N_STATE,
        "extended n_state must equal {N_STATE}"
    );
    debug_assert_eq!(
        state.nonzero_state_indices.len(),
        N_STATE,
        "extended mask must cover all {N_STATE} state dims"
    );

    let fcf = build_fcf(state.n_state);
    let nnz_per_cut = state.nonzero_state_indices.len() + 1;
    let mut batch = build_row_batch(nnz_per_cut);

    c.bench_function("bench_cut_application_with_anticipated", |b| {
        b.iter(|| {
            build_cut_row_batch_into(
                &mut batch,
                black_box(&fcf),
                black_box(0),
                black_box(&state),
                black_box(&[]),
            );
            black_box(batch.values.len());
        });
    });
}

criterion_group!(
    benches,
    bench_cut_application_baseline,
    bench_cut_application_with_anticipated
);
criterion_main!(benches);
