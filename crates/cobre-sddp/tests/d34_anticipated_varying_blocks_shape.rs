//! Case-shape assertions for the D34 deterministic fixture.
//!
//! D34 is the regression backstop for the relocation of the
//! `anticipated_state_out` LP column out of the per-block (`n_blks`-dependent)
//! control region and into the stage-invariant state region. The bug that
//! relocation fixes can only fire when an anticipated commitment **matures at an
//! interior stage whose block count differs from stage 0's** — so this fixture
//! must simultaneously satisfy two shape constraints that no shipped case
//! combined before:
//!
//! 1. at least one anticipated thermal whose `lead_stages` `K_i` matures
//!    **strictly inside** the horizon (`stage + K_i < n_stages`) at an interior
//!    delivery stage, and
//! 2. a per-stage-varying block schedule (block counts differ across stages),
//!    with the maturation stage landing on an off-stage-0 block count.
//!
//! This test pins those two properties so a future edit to the fixture inputs
//! that silently flattens the block schedule, drops the anticipated thermal, or
//! pushes the only commitment's maturation outside the horizon is caught here
//! rather than degrading the `parity_hash_d34` regression to a no-op.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;

fn d34_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/deterministic")
        .join("d34-anticipated-varying-blocks")
}

#[test]
fn d34_combines_anticipated_thermal_with_non_uniform_block_schedule() {
    let system = cobre_io::load_case(&d34_dir()).expect("D34 case must load");

    let block_counts: Vec<usize> = system.stages().iter().map(|s| s.blocks.len()).collect();
    assert_eq!(
        block_counts,
        vec![1, 3, 2],
        "D34 must ship the d33-style non-uniform [1, 3, 2] block schedule; \
         got {block_counts:?}"
    );
    let stage0_blocks = block_counts[0];
    assert!(
        block_counts.iter().any(|&c| c != stage0_blocks),
        "at least one interior stage must differ from stage 0's block count, \
         or the case cannot exercise an off-stage-0 maturation"
    );

    let n_stages = system.stages().len();

    let anticipated: Vec<_> = system
        .thermals()
        .iter()
        .filter_map(|t| t.anticipated_config.map(|cfg| (t.id, cfg.lead_stages)))
        .collect();
    assert!(
        !anticipated.is_empty(),
        "D34 must declare at least one anticipated thermal"
    );

    // A commitment at decision stage `s` matures at `s + K_i` and is active iff
    // `s + K_i < n_stages` (strict). The off-stage-0 maturation requires such a
    // delivery stage to land on an interior block count differing from stage 0's.
    let exercises_off_stage0_maturation = anticipated.iter().any(|&(_, k_i)| {
        let k = k_i as usize;
        (0..n_stages).any(|decision_stage| {
            let delivery_stage = decision_stage + k;
            delivery_stage < n_stages && block_counts[delivery_stage] != stage0_blocks
        })
    });
    assert!(
        exercises_off_stage0_maturation,
        "no anticipated commitment matures strictly inside the horizon at an \
         interior stage whose block count differs from stage 0's; \
         anticipated K_i = {anticipated:?}, block_counts = {block_counts:?}, \
         n_stages = {n_stages} — the case would not exercise the relocated \
         anticipated_state_out column"
    );

    // Pin the K=1 maturation coordinates: decision at stage 0 delivers at stage 1
    // (3 blocks), decision at stage 1 delivers at stage 2 (2 blocks) — both off
    // stage 0's single-block stride.
    let (_, k1) = anticipated
        .iter()
        .find(|&&(_, k)| k == 1)
        .expect("D34's anticipated thermal uses lead_stages = 1");
    assert_eq!(*k1, 1);
    assert_eq!(block_counts[1], 3, "stage-1 delivery lands on 3 blocks");
    assert_eq!(block_counts[2], 2, "stage-2 delivery lands on 2 blocks");
}
