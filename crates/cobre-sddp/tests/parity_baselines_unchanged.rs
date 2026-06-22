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
        "afafd3ebbde7f276cb42b1c50203f67122af3208d76be90437a3f005839ea305",
    ),
    (
        "D02",
        "885d47ceda6daef4e00b024d89637dacdb6971eb3d32eb2dccbe2c3360689a2b",
    ),
    (
        "D03",
        "ed19139d277bb7affa95ada515f0a1e933d6e275fcf70588b26207ca9234d47c",
    ),
    (
        "D04",
        "7834243cd6e85e8fa0728500ee0663752f0c82c98dfb3f70c17a528657916a9e",
    ),
    (
        "D05",
        "d21eec148c510eadc76be65296548e1c6ba3a32f6f324dcf19a86d016b73c9a7",
    ),
    (
        "D06",
        "f618d2a4331f075f5d1aea64b1598aa5569b6086dcab9ff48a750f392b970ca6",
    ),
    (
        "D07",
        "01bbf84d0482a57f0fc35a18a8e24e3fa79e502aca9747b41630f7c554565b68",
    ),
    (
        "D08",
        "143e98dd6ee7fdcb23f2fdea2ce435d370a3e566caea25724b130ecdf542e5d3",
    ),
    (
        "D09",
        "e6f151f5cbdafa61544861b1b2ff6df96887325cdffc0ba8a12b3244bc019c5a",
    ),
    (
        "D10",
        "b43d45e83217d64837bf0f68206a82fb9f9fe32c080abc389bc8c79b15da0417",
    ),
    (
        "D11",
        "189d12201db87e68105e513e25ffda36edf161dada977d82693e5860c31b551d",
    ),
    (
        "D13",
        "7c1c95e542cd358cd524c827d852a616d1d8a906a7f111112704123f22905de5",
    ),
    (
        "D14",
        "932f1c97f15257c048d5131e051af0a174bf168de0ec1f61f6a7d9481160ce87",
    ),
    (
        "D15",
        "16701a7b7f6554794989f4bb0f3ce339693777edef873abfb60471d25112df7f",
    ),
    (
        "D17",
        "c23607d2088fd6901d92c7688a6a28233f598082ac632586bc8c6125396f598a",
    ),
    // Cascade case whose downstream plant declares a `reference_volume`. The
    // reference-volume *default* (0.65) path is covered by the unchanged
    // D05/D06/D07 baselines (none declare a `reference_volume`); D31 exercises the
    // *declared* path. This fast guard only pins the checked-in `D31.sha256`
    // against EXPECTED_HASHES; the full train+sim parity hash for D31 is verified
    // by the slow-gated `parity_hash_d31` (HiGHS and CLP).
    (
        "D31",
        "08bd760f3718196fa9324a56dc08cab8f725365a28267c0218a37ac884ab94b3",
    ),
    // Reversible plant: one pumping station lifts water from the downstream
    // reservoir back up to the upstream one every block, with `flow.min_m3s > 0`
    // forcing a non-degenerate transfer (and matching power draw) on every solve.
    // The pumping column is the first deterministic case to actually participate
    // in the LP, so this hash captures the transfer's effect on both reservoirs'
    // storage trajectories and water values. This guard pins only the HiGHS
    // `D32.sha256`; the CLP `D32.sha256` is verified by the slow-gated CLP
    // `parity_hash_d32`.
    (
        "D32",
        "be72c5faf5881687792dcd4939750906f61b984dc3209c7f8da97269d13a9e29",
    ),
    // Pumping commissioning gating: the D32 reversible plant with an
    // `entry_stage_id`/`exit_stage_id` window so it is active only at stage 1.
    // The omitted stages (0 and 2) shed the forced `flow.min_m3s` transfer, so
    // their storage trajectories diverge from the always-active D32 baseline.
    // This guard pins only the HiGHS `D35.sha256`; the CLP `D35.sha256` is
    // verified by the slow-gated CLP `parity_hash_d35`.
    (
        "D35",
        "41f7a61ff3fa99d7144b37b2614c6d02ac7c771b2e5f29d2be9c9a3102c327d5",
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
