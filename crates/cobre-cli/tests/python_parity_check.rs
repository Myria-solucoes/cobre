//! Integration test: Python parity check.
//!
//! Runs `scripts/ci/check_python_parity.py`, the canonical source-level
//! enforcement of the CLI↔Python output-parity hard rule.

#![allow(clippy::expect_used, clippy::panic, clippy::manual_assert)]

use std::path::PathBuf;
use std::process::Command;

#[test]
fn python_parity_script_passes() {
    if !Command::new("python3")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
    {
        eprintln!("python3 not found; skipping python_parity_script_passes");
        return;
    }

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("repo root must be two levels above CARGO_MANIFEST_DIR")
        .to_path_buf();
    let script = root.join("scripts/ci/check_python_parity.py");
    assert!(
        script.exists(),
        "scripts/ci/check_python_parity.py must exist at {}",
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
