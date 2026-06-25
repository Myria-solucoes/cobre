//! Integration test wrapper for the per-opening MPI parity shell script.
//!
//! `#[ignore]`d so default `cargo test` / `cargo nextest run` do not trigger MPI.
//! To run explicitly:
//!
//! ```text
//! cargo test -p cobre-cli --features mpi -- --ignored per_opening_mpi_parity
//! ```
//!
//! Prerequisites (same as the shell script):
//!   - `mpirun` on `PATH` (`OpenMPI` or `MPICH`)
//!   - `target/release/cobre` built with `--features mpi`
//!   - Python 3 with `pyarrow` installed

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir)
        .parent()
        .expect("crates/ parent must exist")
        .parent()
        .expect("repo root must exist")
        .to_path_buf()
}

#[test]
#[ignore = "requires mpirun and a release build with --features mpi"]
fn per_opening_mpi_parity_d01() {
    let root = repo_root();
    let script = root.join("scripts").join("test_per_opening_mpi_parity.sh");
    let case_dir = root
        .join("examples")
        .join("deterministic")
        .join("d01-thermal-dispatch");

    assert!(
        script.exists(),
        "parity script not found: {}",
        script.display()
    );
    assert!(
        case_dir.exists(),
        "D01 case directory not found: {}",
        case_dir.display()
    );

    let status = Command::new("bash")
        .arg(&script)
        .arg(&case_dir)
        .current_dir(&root)
        .status()
        .expect("failed to launch test_per_opening_mpi_parity.sh");

    assert!(
        status.success(),
        "per-opening MPI parity test failed (exit code {:?}). \
         See target/parity_* directories for the preserved outputs.",
        status.code()
    );
}
