//! Migration-rejection grep gate for the role-(a)/role-(b) indexer split.
//!
//! The role-(a) state-vector concern was extracted onto `StateLayout`, leaving
//! `StageIndexer` a slim role-(b) geometry descriptor. This gate scans the
//! production sources under `src/` and asserts the deleted dual-mode surface does
//! not reappear:
//!
//! - the test-only state-only constructor `StageIndexer::new` is gone (every state
//!   layout is now built through `StateLayout::new`), and
//! - the `with_equipment` wrapper is gone (the single role-(b) constructor is
//!   `with_equipment_and_evaporation`).
//!
//! The forbidden tokens are assembled from character arrays so this gate file does
//! not match itself (the project grep-gate convention).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};

/// Recursively collect every `.rs` file under `dir`.
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
fn slim_indexer_has_no_state_only_or_with_equipment_constructor() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    collect_rs(&src, &mut files);
    assert!(!files.is_empty(), "must scan at least one source file");

    // `StageIndexer::new` — the deleted test-only dual-mode state-only constructor.
    let state_only_ctor = token(&[
        'S', 't', 'a', 'g', 'e', 'I', 'n', 'd', 'e', 'x', 'e', 'r', ':', ':', 'n', 'e', 'w', '(',
    ]);
    // `StageIndexer::with_equipment(` — the deleted wrapper (the surviving
    // constructor is `..._and_evaporation`, which carries a longer suffix and is
    // not matched by this exact-paren token).
    let with_equipment_wrapper = token(&[
        'S', 't', 'a', 'g', 'e', 'I', 'n', 'd', 'e', 'x', 'e', 'r', ':', ':', 'w', 'i', 't', 'h',
        '_', 'e', 'q', 'u', 'i', 'p', 'm', 'e', 'n', 't', '(',
    ]);
    // The surviving constructor's token — used to subtract its `with_equipment`
    // prefix occurrences so the wrapper scan does not flag the real constructor.
    let surviving_ctor = token(&[
        'w', 'i', 't', 'h', '_', 'e', 'q', 'u', 'i', 'p', 'm', 'e', 'n', 't', '_', 'a', 'n', 'd',
        '_', 'e', 'v', 'a', 'p', 'o', 'r', 'a', 't', 'i', 'o', 'n',
    ]);

    let mut offenders: Vec<String> = Vec::new();
    for path in &files {
        let body = std::fs::read_to_string(path).expect("read source file");
        for (lineno, line) in body.lines().enumerate() {
            // Skip the gate's own assembled tokens are not in src/, so no special
            // case is needed; comments are not exempted — the deleted symbols must
            // not be referenced anywhere in production sources.
            if line.contains(&state_only_ctor) {
                offenders.push(format!(
                    "{}:{}: state-only constructor reappeared",
                    path.display(),
                    lineno + 1
                ));
            }
            // `with_equipment(` must not appear except as the prefix of the
            // surviving `with_equipment_and_evaporation(` call. Count the wrapper
            // token and subtract the surviving-constructor occurrences on the line.
            let wrapper_hits = line.matches(&with_equipment_wrapper).count();
            if wrapper_hits > 0 {
                let surviving_hits = line.matches(&surviving_ctor).count();
                if wrapper_hits > surviving_hits {
                    offenders.push(format!(
                        "{}:{}: with_equipment wrapper reappeared",
                        path.display(),
                        lineno + 1
                    ));
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "slim-indexer migration regression — the deleted dual-mode constructors \
         reappeared in production sources:\n{}",
        offenders.join("\n")
    );
}
