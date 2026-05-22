//! Guard test: verify that no parity baseline file was silently modified.
//!
//! Reads all 15 baseline files in a fixed hard-coded order, trims each
//! hex digest, concatenates them, and computes a SHA-256 meta-hash over
//! the concatenation. The resulting meta-hash is compared against a constant
//! committed in this file. Any change to a baseline file will cause this test
//! to fail with a message reporting the observed meta-hash.
//!
//! This test is NOT slow-gated: the total I/O is 15 × 64 bytes and the test
//! runs in milliseconds.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::doc_markdown
)]

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Fixed case list — must NOT include D12 or D16 (absent from baseline set)
// ---------------------------------------------------------------------------

const CASES: &[&str] = &[
    "D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09", "D10", "D11", "D13", "D14",
    "D15", "D17",
];

// ---------------------------------------------------------------------------
// Expected meta-hash — update this constant after the initial bootstrap run
// ---------------------------------------------------------------------------

const EXPECTED_META_HASH: &str = "2f5c980e0bdf6c83d37c776bcd43ed3993a21250461ecb4a2ef653fd393d15be";

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
    let mut hasher = Sha256::new();

    for &case in CASES {
        let path = dir.join(format!("{case}.sha256"));
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("baseline {case} unreadable: {e}"));
        let trimmed = raw.trim();
        assert_eq!(trimmed.len(), 64, "baseline {case} is not 64 hex chars");
        assert!(
            trimmed.chars().all(|c| c.is_ascii_hexdigit()),
            "baseline {case} contains non-hex"
        );
        hasher.update(trimmed.as_bytes());
    }

    let meta = format!("{:x}", hasher.finalize());
    assert_eq!(
        meta, EXPECTED_META_HASH,
        "meta-hash mismatch: baseline changed"
    );
}
