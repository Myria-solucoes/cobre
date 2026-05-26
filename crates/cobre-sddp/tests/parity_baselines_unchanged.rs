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
        "6f754531715e6e574c0a3d96664fecdf650cc6ecb07abde124ea73e49588750f",
    ),
    (
        "D02",
        "70e3e1a268ced563f15dcd7471fe408e1cc407a1aca1723a93a6dd8fe7845797",
    ),
    (
        "D03",
        "4f53e5961ddca1d07a970c006d8eb3f11ee12761aba76d5b632401cafc3fe596",
    ),
    (
        "D04",
        "0fe09e7e7c9a9b4b920a785b6b2ece208d38779d68e22c3e7bc6f0fffe339775",
    ),
    (
        "D05",
        "5592a33c1b26f1907e8154f3d17fd2dcd9e3c6598576ee42e96ef17d01fd0b83",
    ),
    (
        "D06",
        "b1511f8e7b7406ba190699222a52e0383aad25ee1e95d03c1177c0e1e77efac0",
    ),
    (
        "D07",
        "ade7d37e9a018eefc6b34a4ab024cda80fd8d60b70778ba8c0ddaaec0745546b",
    ),
    (
        "D08",
        "69b23609a36d9a67ae63e8d2c466bf955142c2f10aa90131fa870e4c7a570d99",
    ),
    (
        "D09",
        "b459b9dc1b63fd540308445ff1016deee2dc05de872cb0b2381d7b0c74b57254",
    ),
    (
        "D10",
        "0f846604dab0dc09f52e476fd2a837a4613f3d2ebcb71722c84bc4a5295b33a4",
    ),
    (
        "D11",
        "c98e6883311d88245a1bd058ec6a6d81395e649a01294001e7479e83799194da",
    ),
    (
        "D13",
        "037e7bc27e5711085cae9fc8124934fd688ed7b41313b3cf040f1ea9f4126b58",
    ),
    (
        "D14",
        "fd861a800d99ab10cc6520f491d3688e9d7e8c76f8342b7ca1df52e059a22fd8",
    ),
    (
        "D15",
        "5e08b53a627715aee52efa23b6a06a3db541bcc520cfdad49fcbac720830aade",
    ),
    (
        "D17",
        "0a7593b638dad98378a19c939658869d09ba5d09c1fc1922b112b0314fe2b2ec",
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
