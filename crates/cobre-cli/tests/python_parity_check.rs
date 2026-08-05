//! Integration test: Python parity check.
//!
//! `scripts/ci/check_python_parity.py` parses both `cobre-cli/src/commands/run.rs`
//! and `cobre-python/src/run.rs` for `cobre_io::write_*` calls and other output
//! helpers and asserts the two sets are identical — the canonical source-level
//! enforcement of the CLI↔Python output-parity hard rule.

#![allow(clippy::expect_used, clippy::panic, clippy::manual_assert)]

use std::path::{Path, PathBuf};
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

fn read_rs_recursive(dir: &Path) -> String {
    let mut buf = String::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for entry in std::fs::read_dir(&d).expect("read_dir must succeed") {
            let path = entry.expect("dir entry must be readable").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                buf.push_str(&std::fs::read_to_string(&path).expect("rs file must read"));
            }
        }
    }
    buf
}

/// The single-owner `write_scenario_summary` serializer must be called from BOTH
/// simulation write paths, so `simulation/scenario_summary.parquet` is produced
/// identically by the CLI and the Python bindings. The generic parity script does
/// not track bare-imported `cobre_io` run-level writers (this one and
/// `write_paths`), so this asserts the call sites directly. A `use` import carries
/// no trailing `(`, so matching `write_scenario_summary(` catches only the call.
#[test]
fn both_simulation_write_sites_call_write_scenario_summary() {
    let root = repo_root();
    let cli = read_rs_recursive(&root.join("crates/cobre-cli/src/commands/run"));
    let python =
        std::fs::read_to_string(root.join("crates/cobre-python/src/run.rs")).expect("run.rs reads");

    let needle = "write_scenario_summary(";
    assert!(
        cli.contains(needle),
        "CLI simulation write path must call write_scenario_summary"
    );
    assert!(
        python.contains(needle),
        "Python simulation write path must call write_scenario_summary"
    );
}
