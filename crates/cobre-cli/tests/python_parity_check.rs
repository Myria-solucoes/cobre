//! Integration tests: Python parity checks.
//!
//! - `python_parity_script_passes`: invokes `python3
//!   scripts/check_python_parity.py --max 0` against the repo root.
//!   Skipped if `python3` is unavailable.
//! - `populated_anticipated_columns_present_in_python_run_rs`: structural
//!   check that `crates/cobre-python/src/run.rs` constructs a
//!   `SimulationParquetWriter`, ensuring the `is_anticipated`,
//!   `anticipated_decision_mw`, and `anticipated_committed_mw` columns
//!   are written by the Python binding path.
//! - `populated_state_dictionary_present_in_python_run_rs`: structural
//!   check that `crates/cobre-python/src/run.rs` calls
//!   `write_training_results`, which transitively emits
//!   `training/dictionaries/state_dictionary.json` (including the
//!   `anticipated_state` slot entries).

#![allow(clippy::expect_used, clippy::panic, clippy::manual_assert)]

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root must be two levels above CARGO_MANIFEST_DIR")
        .to_path_buf()
}

fn python3_available() -> bool {
    Command::new("python3")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

#[test]
fn python_parity_script_passes() {
    if !python3_available() {
        eprintln!("python3 not found; skipping python_parity_script_passes");
        return;
    }

    let root = repo_root();
    let script = root.join("scripts/check_python_parity.py");
    assert!(
        script.exists(),
        "scripts/check_python_parity.py must exist at {}",
        script.display()
    );

    let output = Command::new("python3")
        .arg(&script)
        .arg("--max")
        .arg("0")
        .arg("--root")
        .arg(&root)
        .output()
        .expect("failed to invoke python3");

    if !output.status.success() {
        panic!(
            "Python parity check failed.\n--- stdout ---\n{}\n--- stderr ---\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

/// Structural check: `crates/cobre-python/src/run.rs` must construct a
/// `SimulationParquetWriter`, which is the single writer that emits the
/// `is_anticipated`, `anticipated_decision_mw`, and `anticipated_committed_mw`
/// columns. If a future refactor removes this call from the Python path, the
/// populated columns will silently disappear from Python outputs.
#[test]
fn populated_anticipated_columns_present_in_python_run_rs() {
    let root = repo_root();
    let python_run = root.join("crates/cobre-python/src/run.rs");
    let contents = std::fs::read_to_string(&python_run)
        .unwrap_or_else(|e| panic!("cobre-python/src/run.rs must be readable: {e}"));
    assert!(
        contents.contains("SimulationParquetWriter::new"),
        "crates/cobre-python/src/run.rs must construct a SimulationParquetWriter — \
         the anticipated thermal columns (is_anticipated, anticipated_decision_mw, \
         anticipated_committed_mw) will not be written by the Python path otherwise"
    );
}

/// Structural check: `crates/cobre-python/src/run.rs` must call
/// `write_training_results`, which transitively emits
/// `training/dictionaries/state_dictionary.json` (via `write_results` →
/// `write_dictionaries` → `write_state_dictionary_json`). If a future refactor
/// removes this call from the Python path, the `anticipated_state` slot
/// entries will silently disappear from the Python state dictionary output.
#[test]
fn populated_state_dictionary_present_in_python_run_rs() {
    let root = repo_root();
    let python_run = root.join("crates/cobre-python/src/run.rs");
    let contents = std::fs::read_to_string(&python_run)
        .unwrap_or_else(|e| panic!("cobre-python/src/run.rs must be readable: {e}"));
    assert!(
        contents.contains("write_training_results"),
        "crates/cobre-python/src/run.rs must call \
         cobre_io::write_training_results — the anticipated_state entries \
         in training/dictionaries/state_dictionary.json will not be written \
         by the Python path otherwise"
    );
}
