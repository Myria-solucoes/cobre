//! Realistic-scale determinism tests for the cut-selection kernel.
//!
//! Verifies that `CutSelectionStrategy::select` (which forwards to
//! the GEMM-based kernel) produces bit-identical `CutActivityUpdates`
//! across thread counts {1, 16, 96} at:
//! - Aggregated sizing (K=945, M=384, D=155) — convertido iter-4 scale.
//! - Disaggregated sizing (K=1000, M=384, D=2080) — production
//!   target. Gated behind `slow-tests` feature.
//!
//! Note on host capacity: the disaggregated test requests rayon pools
//! of size {1, 16, 96}. On hosts with fewer than 96 logical CPUs (e.g.
//! GitHub `ubuntu-latest` runners with 2–4 cores) rayon's worker count
//! is capped by the OS, which is acceptable here — the assertions
//! compare byte-identical bitmaps across whatever thread counts rayon
//! actually instantiates.

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::float_cmp,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::too_many_lines,
    )
)]

use cobre_sddp::cut::CutPool;
use cobre_sddp::cut_selection::{CutActivityUpdates, CutSelectionStrategy};

/// Splitmix64 PRNG — same algorithm used in
/// `crates/cobre-solver/examples/audit_mm_dispatch.rs` and
/// `crates/cobre-sddp/examples/probe_k_disaggregated.rs`. Inline so
/// no external crate is needed.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn fill_f64(buf: &mut [f64], seed: u64) {
    let mut state = seed;
    for slot in buf.iter_mut() {
        let r = splitmix64(&mut state);
        let bits = (r >> 12) & ((1u64 << 52) - 1);
        *slot = f64::from_bits((1023u64 << 52) | bits) - 1.5;
    }
}

fn make_pool(k: usize, d: usize, seed: u64) -> CutPool {
    let mut pool = CutPool::new(k, d, 1, 0);
    let mut state = seed;
    for slot in 0..k {
        let r = splitmix64(&mut state);
        let bits = (r >> 12) & ((1u64 << 52) - 1);
        let intercept = f64::from_bits((1023u64 << 52) | bits) - 1.5;
        let mut coeffs = vec![0.0_f64; d];
        fill_f64(&mut coeffs, state.wrapping_add(slot as u64));
        pool.add_cut(0, slot as u32, intercept, &coeffs);
    }
    // Force all cuts eligible: iteration_generated < current_iteration
    // (current_iteration in the tests below is 5).
    for slot in 0..k {
        pool.metadata[slot].iteration_generated = 1;
    }
    pool
}

fn make_states(m: usize, d: usize, seed: u64) -> Vec<f64> {
    let mut buf = vec![0.0_f64; m * d];
    fill_f64(&mut buf, seed);
    buf
}

fn run_in_pool<F>(threads: usize, f: F) -> CutActivityUpdates
where
    F: FnOnce() -> CutActivityUpdates + Send,
{
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon thread pool must build");
    pool.install(f)
}

#[test]
fn select_for_stage_deterministic_aggregated_scale() {
    const K: usize = 945;
    const D: usize = 155;
    const M: usize = 384;

    let pool = make_pool(K, D, 0xAAAA_BBBB_CCCC_DDDD);
    let states = make_states(M, D, 0x1111_2222_3333_4444);

    let strategy = CutSelectionStrategy::Lml1 {
        check_frequency: 1,
        tie_tolerance: 1e-10,
    };

    let r1 = run_in_pool(1, || strategy.select(&pool, &states, 5));
    let r16 = run_in_pool(16, || strategy.select(&pool, &states, 5));
    let r96 = run_in_pool(96, || strategy.select(&pool, &states, 5));

    assert_eq!(
        r1.updates, r16.updates,
        "deactivations differ between 1- and 16-thread runs at aggregated scale"
    );
    assert_eq!(
        r1.updates, r96.updates,
        "deactivations differ between 1- and 96-thread runs at aggregated scale"
    );
    assert_eq!(
        r1.reactivations, r16.reactivations,
        "reactivations differ between 1- and 16-thread runs at aggregated scale"
    );
    assert_eq!(
        r1.reactivations, r96.reactivations,
        "reactivations differ between 1- and 96-thread runs at aggregated scale"
    );
}

/// Smaller medium-scale fixture exercising Level1 and Dominated to
/// confirm determinism is not Lml1-specific. K=200, D=155, M=64 —
/// runs in <100 ms total so it stays in the default suite.
#[test]
fn select_for_stage_deterministic_level1_and_dominated_medium() {
    const K: usize = 200;
    const D: usize = 155;
    const M: usize = 64;

    let pool = make_pool(K, D, 0x0001_0002_0003_0004);
    let states = make_states(M, D, 0x0005_0006_0007_0008);

    for strategy in [
        CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        },
        CutSelectionStrategy::Dominated {
            threshold: 0.01,
            check_frequency: 1,
        },
    ] {
        let r1 = run_in_pool(1, || strategy.select(&pool, &states, 5));
        let r8 = run_in_pool(8, || strategy.select(&pool, &states, 5));
        assert_eq!(
            r1.updates, r8.updates,
            "deactivations differ for strategy {strategy:?}"
        );
        assert_eq!(
            r1.reactivations, r8.reactivations,
            "reactivations differ for strategy {strategy:?}"
        );
    }
}

/// Disaggregated-scale test: K=1000 (placeholder until
/// Epic 01 ticket-003's K probe lands), M=384, D=2080.
/// ~1.2 billion FMA per call × 3 pools = ~6 seconds in release on
/// modern `x86_64` (per design §6 sizing model). Gated behind
/// `slow-tests` per project rule.
#[test]
#[cfg_attr(
    not(feature = "slow-tests"),
    ignore = "slow: run with --features slow-tests"
)]
fn select_for_stage_deterministic_disaggregated_scale() {
    // TODO(Epic 01 ticket-003): replace K=1000 with the measured
    // disaggregated K value once the probe report lands in
    // docs/design/cut-selection-parallelism-redesign.md §14.3.
    const K: usize = 1000;
    const D: usize = 2080;
    const M: usize = 384;

    let pool = make_pool(K, D, 0xFEED_FACE_CAFE_BABE);
    let states = make_states(M, D, 0xDEAD_BEEF_BAAD_F00D);

    let strategy = CutSelectionStrategy::Lml1 {
        check_frequency: 1,
        tie_tolerance: 1e-10,
    };

    let r1 = run_in_pool(1, || strategy.select(&pool, &states, 5));
    let r16 = run_in_pool(16, || strategy.select(&pool, &states, 5));
    let r96 = run_in_pool(96, || strategy.select(&pool, &states, 5));

    assert_eq!(
        r1.updates, r16.updates,
        "deactivations differ between 1- and 16-thread runs at disaggregated scale"
    );
    assert_eq!(
        r1.updates, r96.updates,
        "deactivations differ between 1- and 96-thread runs at disaggregated scale"
    );
    assert_eq!(
        r1.reactivations, r16.reactivations,
        "reactivations differ between 1- and 16-thread runs at disaggregated scale"
    );
    assert_eq!(
        r1.reactivations, r96.reactivations,
        "reactivations differ between 1- and 96-thread runs at disaggregated scale"
    );
}
