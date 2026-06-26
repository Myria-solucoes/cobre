//! Deletion-regression grep gate for the removed indexer types.
//!
//! The role-(b) geometry descriptor `StageIndexer` and its `EquipmentCounts`
//! constructor input were deleted: the state-vector concern lives on
//! `StateLayout`, the non-state study shape on `StudyDimensions`, and the
//! per-stage equipment geometry on `StageLayout`/`StageGeometry`. This gate scans
//! the production sources under `src/` and asserts those deleted types do not
//! reappear — a regression guard that the deletion stays deleted.
//!
//! The forbidden tokens are assembled from character arrays so this gate file
//! does not match itself (the project grep-gate convention).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read_dir src") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_dir() {
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// Assemble a forbidden token from chars so the gate does not self-match.
fn token(chars: &[char]) -> String {
    chars.iter().collect()
}

#[test]
fn deleted_indexer_types_stay_deleted() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rs(&src, &mut files);
    assert!(!files.is_empty(), "must scan at least one source file");

    // `StageIndexer` — the deleted role-(b) geometry descriptor type.
    let stage_indexer = token(&['S', 't', 'a', 'g', 'e', 'I', 'n', 'd', 'e', 'x', 'e', 'r']);
    // `EquipmentCounts` — the deleted constructor input bag.
    let equipment_counts = token(&[
        'E', 'q', 'u', 'i', 'p', 'm', 'e', 'n', 't', 'C', 'o', 'u', 'n', 't', 's',
    ]);

    let mut offenders: Vec<String> = Vec::new();
    for path in &files {
        let body = std::fs::read_to_string(path).expect("read source file");
        for (lineno, line) in body.lines().enumerate() {
            if line.contains(&stage_indexer) {
                offenders.push(format!(
                    "{}:{}: deleted type StageIndexer reappeared",
                    path.display(),
                    lineno + 1
                ));
            }
            if line.contains(&equipment_counts) {
                offenders.push(format!(
                    "{}:{}: deleted type EquipmentCounts reappeared",
                    path.display(),
                    lineno + 1
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "deletion regression — a deleted indexer type reappeared in production \
         sources (the role-(b) geometry now lives on StageLayout/StageGeometry, \
         the state vector on StateLayout, the study shape on StudyDimensions):\n{}",
        offenders.join("\n")
    );
}
