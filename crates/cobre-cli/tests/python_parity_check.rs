//! Integration test: Python parity check.
//!
//! `python_parity_script_passes` invokes `python3
//! scripts/check_python_parity.py --max 0` against the repo root. The
//! script parses both `crates/cobre-cli/src/commands/run.rs` and
//! `crates/cobre-python/src/run.rs` for `cobre_io::write_*` calls and
//! other output helpers and asserts the set is identical — the
//! canonical source-level parity contract.
//!
//! Earlier revisions of this file carried two `contents.contains(...)`
//! grep tests targeted at the anticipated-thermals output calls. Those
//! tests were subset checks of what the script already covers, and any
//! call site that happened to mention the substring in a comment or
//! string literal would satisfy the assertion. Per assessment finding
//! F3-005 the redundant grep tests are removed; the script is the
//! canonical check.
//!
//! A future behavioural parity test — running cobre CLI and cobre-python
//! against the same fixture and asserting byte-for-byte parquet
//! equality — is tracked separately. Until then, the script-based
//! parity check is the hard rule's enforcement point.

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
