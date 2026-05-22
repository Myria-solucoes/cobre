//! Guard test: verify that no parity baseline file was silently modified.
//!
//! Reads each baseline file in a fixed hard-coded order, trims its hex
//! digest, and compares it byte-for-byte against the expected value pinned
//! in [`EXPECTED_HASHES`]. Any change to a baseline file will cause this
//! test to fail with a message pointing at the offending case.
//!
//! The original implementation hashed all baselines into a single
//! meta-hash via `sha2::Sha256`. That was a double-hash (the baseline
//! files are already SHA-256 digests) and gave weak diagnostics — a
//! meta-hash mismatch tells you *something* changed but not which file.
//! The per-file table here uses plain string equality and identifies the
//! offending case directly. The `sha2` crate is still a dev-dependency
//! because `parity_hash_d01_d15.rs` legitimately uses it to compute the
//! baselines themselves from training output.
//!
//! This test is NOT slow-gated: the total I/O is 15 × 64 bytes and the
//! test runs in milliseconds.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::doc_markdown
)]

use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Expected hex digests — one entry per baseline file
// ---------------------------------------------------------------------------
//
// Updating this table: re-run the relevant deterministic case end-to-end
// (e.g., `just regression-baseline D06`) and paste the new hash here, in the
// commit that intentionally changes the baseline.

const EXPECTED_HASHES: &[(&str, &str)] = &[
    (
        "D01",
        "93a6d643a1e0aad4b2d23260c05c7f2fec55f4084948a84ab4ac48dc8dc974c2",
    ),
    (
        "D02",
        "02057f4df2b03630de34039dc8274caf22e27eedff6668268f11727edccbfa4d",
    ),
    (
        "D03",
        "dc80579fd558423040e5e47a4827cd48cb843dc4d8ce24475ce4ca9525ff1e4e",
    ),
    (
        "D04",
        "41a3a7a6dc9a918276ba5e6a1fd72f235640566ea7cfed93f4402c242363ceb9",
    ),
    (
        "D05",
        "7a887f1762ed147c65442c6e6ddec4e8db6b5309b7a2cfd2440a64529f7cd8b3",
    ),
    (
        "D06",
        "7e089dbeac378cc286721e7fad9dd13d707fe8421d1248f2481e8716a815742c",
    ),
    (
        "D07",
        "5b39ddf297c6f32c61578b38d3121d66b7e2539b13cb81f093e1bd25a6872f28",
    ),
    (
        "D08",
        "d5dd47c2a9946ed5dd6d24241e0b5100b1b8123215eb2ab76cdd68f3ee2d3297",
    ),
    (
        "D09",
        "4c4ae3bd684b0bb770267e93e7b9fd5f75d067b91a054974d14e83c7432bf919",
    ),
    (
        "D10",
        "db4f9fcdfc76533cff9ee77ca51d551d354db40da44a0207bff57c0eebb50d00",
    ),
    (
        "D11",
        "4f5fe3be2084e07ecd401f85d446ae73ff882e58838cd18464337a566f4c5bd3",
    ),
    (
        "D13",
        "30b08bcd0a08a3b2a30bfa401725e452c1b2b5c258a8d590884b777b61e04abe",
    ),
    (
        "D14",
        "a4e9c92e924eff45f969f604c67e38ed3ff05f9c995297bbbd31fa33e50acbc0",
    ),
    (
        "D15",
        "d437f0b2468cc74fbff832238f7021659976694f6d894c0d077a70d98ac79804",
    ),
    (
        "D17",
        "9bec6d92849a8ade7cd8682274ffd78c61ef93f1c90723fe1a95d69ae761ebaf",
    ),
];

// ---------------------------------------------------------------------------
// Helper: baseline directory path
// ---------------------------------------------------------------------------

fn baseline_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parity_baselines")
}

// ---------------------------------------------------------------------------
// Guard test
// ---------------------------------------------------------------------------

#[test]
fn parity_baselines_have_not_changed() {
    let dir = baseline_dir();
    let mut drifted: Vec<(&str, String, &str)> = Vec::new();

    for &(case, expected) in EXPECTED_HASHES {
        let path = dir.join(format!("{case}.sha256"));
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("baseline {case} unreadable: {e}"));
        let trimmed = raw.trim();
        assert_eq!(trimmed.len(), 64, "baseline {case} is not 64 hex chars");
        assert!(
            trimmed.chars().all(|c| c.is_ascii_hexdigit()),
            "baseline {case} contains non-hex"
        );
        if trimmed != expected {
            drifted.push((case, trimmed.to_string(), expected));
        }
    }

    assert!(
        drifted.is_empty(),
        "parity baselines drifted ({} of {} cases):\n{}",
        drifted.len(),
        EXPECTED_HASHES.len(),
        drifted
            .iter()
            .map(|(case, actual, expected)| format!("  {case}: expected {expected}, got {actual}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
