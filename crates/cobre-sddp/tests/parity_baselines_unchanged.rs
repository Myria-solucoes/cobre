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
#![cfg(feature = "highs")]

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
        "274974027bf192c228697e40c0fb3da30e29606885caf27d8ec38e41e8f65f1f",
    ),
    (
        "D03",
        "66b60e4138c51346f76ec1f062fcfc6f194014a47b901c4f6254bf7b33364dc7",
    ),
    (
        "D04",
        "0ac2a66bdf85b8f6b566720fc6492f82820d84f144b474d5ac186e7d40837327",
    ),
    (
        "D05",
        "a0f4602ec0e2f2872456a5685622ef877c18947072c9e22cdbc81ff9785803db",
    ),
    (
        "D06",
        "4d2631c0f6aeb966f181b213bbf952064e6efca20d7e2a2c0ddf2be5e8d0bace",
    ),
    (
        "D07",
        "8181b12c2e0d527cf246cb77dfab169b4c39e9875c7adf3d43e0ebafa7b0cb5a",
    ),
    (
        "D08",
        "9a6784527cdc180ac3b03d2d29f4d436e19bf10c55924dc6893996a5fb344de5",
    ),
    (
        "D09",
        "b459b9dc1b63fd540308445ff1016deee2dc05de872cb0b2381d7b0c74b57254",
    ),
    (
        "D10",
        "82702f68338b80e34d0690d70b017738e5019ffd2cf71e63201c031e82d41347",
    ),
    (
        "D11",
        "6ac8e25e9fe811c70390c3e148b1f9f1dab897fc2b75389aed40a1628315d943",
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
        "01033bd40ad2d858eb0c03b2efd462ea43c6854e1a557b274b8a6e9001278a9a",
    ),
    // Cascade case whose downstream plant declares a `reference_volume`. The
    // reference-volume *default* (0.65) path is covered by the unchanged
    // D05/D06/D07 baselines (none declare a `reference_volume`); D31 exercises the
    // *declared* path. This fast guard only pins the checked-in `D31.sha256`
    // against EXPECTED_HASHES; the full train+sim parity hash for D31 is verified
    // by the slow-gated `parity_hash_d31` (HiGHS and CLP).
    (
        "D31",
        "29ef7566025ab1938b2d66bfbc2b4139c258ff940d0ed79cce57aedc51bd4086",
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
