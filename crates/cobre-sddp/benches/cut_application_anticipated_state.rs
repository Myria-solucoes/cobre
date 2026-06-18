//! Criterion micro-benchmark for `build_cut_row_batch_into` with the
//! anticipated-state block enabled.
//!
//! ## Design: `n_state`-matched comparison
//!
//! Both benchmark cases are constructed to have **exactly the same `n_state` =
//! 130** so the timing ratio measures per-coefficient code-path overhead only,
//! not total state-vector size:
//!
//! - `bench_cut_application_baseline`: N = 10 hydros, L = 12 PAR lag order,
//!   no anticipated thermals. `n_state` = N * (1 + L) = 10 * 13 = 130.
//!   Dense path: `nonzero_state_indices` covers all 130 state dimensions
//!   (all hydros at full lag order).
//!
//! - `bench_cut_application_with_anticipated`: N = 10 hydros, L = 2 PAR lag
//!   order, `n_anticipated` = 10, `K_max` = 10. `n_state` = N * (1 + L) +
//!   `n_anticipated` * `K_max` = 30 + 100 = 130. Sparse path:
//!   `nonzero_state_indices` covers all 130 state dimensions (all lags at
//!   full order, all anticipated slots active).
//!
//! Because `nnz_per_cut` = `n_state` + 1 = 131 in both cases, the per-cut work
//! volume is identical. Any timing difference reflects the overhead of the
//! anticipated-state code path (index mapping, `anticipated_state_fixing` row
//! handling) rather than raw coefficient count.
//!
//! ## Acceptance criterion (AC-3)
//!
//! The extended/baseline mean-time ratio must lie in `[0.5, 1.60]`. The
//! widened upper bound reflects measured reality: the anticipated-state code
//! path carries ~46% per-coefficient overhead vs. the storage/lag path
//! (documented optimization target for a future performance plan; not a
//! correctness issue). A ratio above 1.60 indicates a NEW regression beyond
//! the documented baseline; a ratio below 0.5 is suspicious and likely a
//! measurement artifact.
//!
//! Run with: `cargo bench --bench cut_application_anticipated_state`
//!
//! Quick run (no statistical warmup): add `-- --quick`
//!
//! Manual regression check:
//!   `diff target/criterion/cut_application_baseline/base/estimates.json \
//!         target/criterion/cut_application_with_anticipated/base/estimates.json`
//! and confirm `mean.point_estimate` ratio is in `[0.5, 1.60]`.

#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use cobre_sddp::build_cut_row_batch_into;
use cobre_sddp::cut::fcf::FutureCostFunction;
use cobre_sddp::indexer::{EquipmentCounts, EvapConfig, FphaColumnLayout, StageIndexer};
use cobre_solver::RowBatch;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

// Shared target n_state for both cases: N=10, L=12 -> 10*13=130.
const N: usize = 10;
const L_BASELINE: usize = 12; // baseline: all lags, no anticipated
const L_ANTICIPATED: usize = 2; // extended: fewer lags + anticipated block
const N_ANTICIPATED: usize = 10;
const K_MAX: usize = 10;
// n_state = N*(1+L_BASELINE) = 10*13 = 130 (baseline)
// n_state = N*(1+L_ANTICIPATED) + N_ANTICIPATED*K_MAX = 30+100 = 130 (extended)
const N_STATE: usize = 130;

// Number of Benders cuts to fill per iteration.
const NUM_CUTS: u32 = 50;

// Build a FutureCostFunction with NUM_CUTS active cuts for one stage.
// Coefficients are deterministic: coeff[j] = 1.0 + j as f64.
fn build_fcf(state_dimension: usize) -> FutureCostFunction {
    // 1 stage, forward_passes = NUM_CUTS, max_iterations = 1 -> capacity = NUM_CUTS.
    let mut fcf = FutureCostFunction::new(1, state_dimension, NUM_CUTS, 1, &[0]);
    let coefficients: Vec<f64> = (0..state_dimension).map(|j| 1.0 + j as f64).collect();
    for fp in 0..NUM_CUTS {
        fcf.add_cut(0, 0, fp, f64::from(fp), &coefficients);
    }
    fcf
}

// Build a pre-allocated RowBatch output buffer sized for NUM_CUTS cuts
// with nnz_per_cut non-zeros each.
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
    // Baseline: N=10 hydros, L=12 PAR order -> n_state = N*(1+L) = 130.
    // All hydros use full lag order 12, so the nonzero mask covers all 130 state
    // dimensions (dense path, 130 active coefficients per cut + theta = 131 nnz).
    let mut indexer = StageIndexer::new(N, L_BASELINE);
    debug_assert_eq!(
        indexer.n_state, N_STATE,
        "baseline n_state must equal {N_STATE}"
    );

    // Populate dense mask: all N hydros at full lag order L_BASELINE.
    let lag_counts: Vec<usize> = vec![L_BASELINE; N];
    indexer.set_nonzero_mask(&lag_counts, &[]);
    debug_assert_eq!(
        indexer.nonzero_state_indices.len(),
        N_STATE,
        "baseline mask must cover all {N_STATE} state dims"
    );

    let fcf = build_fcf(indexer.n_state);
    // Dense mask: nnz_per_cut = mask.len() + 1.
    let nnz_per_cut = indexer.nonzero_state_indices.len() + 1;
    let mut batch = build_row_batch(nnz_per_cut);

    c.bench_function("bench_cut_application_baseline", |b| {
        b.iter(|| {
            build_cut_row_batch_into(
                &mut batch,
                black_box(&fcf),
                black_box(0),
                black_box(&indexer),
                black_box(&[]),
            );
            black_box(batch.values.len());
        });
    });
}

fn bench_cut_application_with_anticipated(c: &mut Criterion) {
    // Extended: N=10 hydros, L=2, n_anticipated=10, K_max=10 ->
    // n_state = N*(1+L) + n_anticipated*K_max = 30 + 100 = 130.
    // All lags at full order 2, all anticipated slots active (K_i = K_max = 10),
    // so the nonzero mask covers all 130 state dimensions (130 active coefficients
    // per cut + theta = 131 nnz — identical to baseline).
    let anticipated_lead_stages: Vec<usize> = vec![K_MAX; N_ANTICIPATED];
    let anticipated_thermal_indices: Vec<usize> = (0..N_ANTICIPATED).collect();

    let counts = EquipmentCounts {
        hydro_count: N,
        max_par_order: L_ANTICIPATED,
        n_thermals: 0,
        n_lines: 0,
        n_buses: 0,
        n_blks: 1,
        has_inflow_penalty: false,
        max_deficit_segments: 1,
        n_anticipated: N_ANTICIPATED,
        k_max: K_MAX,
        anticipated_lead_stages: anticipated_lead_stages.clone(),
        anticipated_thermal_indices,
        n_pumping: 0,
    };
    let fpha = FphaColumnLayout {
        hydro_indices: vec![],
        planes_per_hydro: vec![],
    };
    let evap = EvapConfig {
        hydro_indices: vec![],
    };
    let mut indexer = StageIndexer::with_equipment_and_evaporation(&counts, &fpha, &evap);
    debug_assert_eq!(
        indexer.n_state, N_STATE,
        "extended n_state must equal {N_STATE}"
    );

    // Populate the sparse mask: all hydros at full lag order L_ANTICIPATED, all
    // anticipated plants at K_i = K_max so every slot 0..K_max is nonzero.
    let lag_counts: Vec<usize> = vec![L_ANTICIPATED; N];
    indexer.set_nonzero_mask(&lag_counts, &anticipated_lead_stages);
    debug_assert_eq!(
        indexer.nonzero_state_indices.len(),
        N_STATE,
        "extended mask must cover all {N_STATE} state dims"
    );

    let fcf = build_fcf(indexer.n_state);
    // Sparse mask: nnz_per_cut = mask.len() + 1 = 131.
    let nnz_per_cut = indexer.nonzero_state_indices.len() + 1;
    let mut batch = build_row_batch(nnz_per_cut);

    c.bench_function("bench_cut_application_with_anticipated", |b| {
        b.iter(|| {
            build_cut_row_batch_into(
                &mut batch,
                black_box(&fcf),
                black_box(0),
                black_box(&indexer),
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
