//! Fast guard: the committed parity baselines match the digests pinned in
//! [`EXPECTED_HASHES`], so a baseline cannot change silently.
//!
//! This is the third pin of every HiGHS baseline (the other two are the
//! checked-in `parity_baselines/<case>.sha256` file and the slow-gated
//! `parity_hash_*` end-to-end run); a re-baseline must update all three together
//! or this guard rots silently. Per-file string equality (not a meta-hash) names
//! the offending case directly. NOT slow-gated — each baseline is one 64-byte
//! digest.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::doc_markdown
)]
#![cfg(feature = "highs")]

use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Expected hex digests — one entry per HiGHS baseline file
// ---------------------------------------------------------------------------
//
// Each entry pins only the HiGHS `<case>.sha256`; the CLP baseline is verified
// by the slow-gated CLP `parity_hash_<case>`. To intentionally change a
// baseline, re-run the case end-to-end (e.g. `just regression-baseline D06`) and
// paste the new hash here in the same commit.

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
    (
        "D31",
        "08bd760f3718196fa9324a56dc08cab8f725365a28267c0218a37ac884ab94b3",
    ),
    (
        "D32",
        "be72c5faf5881687792dcd4939750906f61b984dc3209c7f8da97269d13a9e29",
    ),
    (
        "D35",
        "ebd271e4504810723f9a380130d5ca20662ddfd88194c5418555b26d3a80d977",
    ),
    (
        "D36",
        "ea73313186c94b872aa8a755c1c00d9537688be74971dd1bebb6bb0d35376e62",
    ),
    (
        "D37",
        "c709a7433e5a2f1ceac3bb0030806ccdd85ee3ecbc2578e66a156dd3bfce7bd0",
    ),
    (
        "D38",
        "7e02e0cfd3d7cbf27ec4d1939a9e6957e52eecaa3e20ccdb2ef2be03426de352",
    ),
    (
        "D39",
        "7a29931dcd7e678c3e0ce83820508c97c9419e620669f98ae25b17292c8ad7e8",
    ),
    (
        "D40",
        "e9e2508363c3d2cd6524fc822849f8d35ca76a3909f51f48fd7325d53c6d1968",
    ),
    (
        "D41",
        "a6093e62935c6ace5fadb9970de58f390f46602bdaddbe85870dc02a0e30b988",
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
