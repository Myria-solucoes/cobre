//! CLP-only end-to-end smoke test.
//!
//! Guarantees the clp-only build (`--no-default-features --features clp`) ships
//! at least one runnable integration test, not just `#[cfg(test)]` unit tests.
//!
//! The fixture is duplicated from the `#[cfg(test)]` unit tests because
//! integration tests cannot reach module internals.
#![cfg(feature = "clp")]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::float_cmp,
        clippy::panic
    )
)]

use cobre_solver::{ClpSolver, SolverInterface, StageTemplate};

/// Builds the shared SS1.1 fixture LP (3 cols, 2 equality rows).
///
/// Optimal solution: objective `100.0`, primals `(6.0, 0.0, 2.0)`.
fn make_fixture_stage_template() -> StageTemplate {
    StageTemplate {
        num_cols: 3,
        num_rows: 2,
        num_nz: 3,
        col_starts: vec![0_i32, 2, 2, 3],
        row_indices: vec![0_i32, 1, 1],
        values: vec![1.0, 2.0, 1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![10.0, f64::INFINITY, 8.0],
        objective: vec![0.0, 1.0, 50.0],
        row_lower: vec![6.0, 14.0],
        row_upper: vec![6.0, 14.0],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 1,
        n_hydro: 1,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

#[test]
fn clp_only_load_model_and_solve() {
    let mut solver = ClpSolver::new().expect("CLP init");
    let template = make_fixture_stage_template();

    solver.load_model(&template);
    let sol = solver.solve(None).expect("solve");

    assert!(
        (sol.objective - 100.0).abs() < 1e-9,
        "expected objective = 100.0, got {}",
        sol.objective
    );

    let primals = &sol.primal;
    assert!(
        (primals[0] - 6.0).abs() < 1e-9,
        "expected x0 = 6.0, got {}",
        primals[0]
    );
    assert!(
        (primals[1] - 0.0).abs() < 1e-9,
        "expected x1 = 0.0, got {}",
        primals[1]
    );
    assert!(
        (primals[2] - 2.0).abs() < 1e-9,
        "expected x2 = 2.0, got {}",
        primals[2]
    );
}
