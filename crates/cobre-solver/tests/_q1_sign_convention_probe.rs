//! `Q1` microbenchmark — sign convention of `reduced_cost` for fixed-at-bound columns.
//!
//! For the LP simplification plan (column-bound state-fixing): we need to know
//! whether `HiGHS` reports `reduced_cost` for a column with `lb == ub` such that
//! it numerically matches the row dual of the equivalent equality row.
//!
//! Two equivalent LPs are constructed:
//! - **LP-R**: `x1` is free in `[0, 10]`, with an equality row `x1 == 7`.
//! - **LP-C**: `x1` is fixed via column bounds `lb = ub = 7`; no equality row.
//!
//! Both LPs share:
//! - `x2` free in `[0, 10]`
//! - `x3` free in `[0, 10]`
//! - Objective: `min 5*x1 + 2*x2 + 3*x3`
//! - Constraint: `x2 + x3 >= 4` (just to keep them non-trivial)
//!
//! Expected solution: `x1 = 7`, `x2 = 4`, `x3 = 0`, obj = 35 + 8 = 43.
//!
//! What we want to learn:
//! - LP-R: `dual[equality_row]` — sign and magnitude.
//! - LP-C: `reduced_cost[x1]` — sign and magnitude.
//! - Do they match?
//!
//! Run with: `cargo test -p cobre-solver --test _q1_sign_convention_probe -- --nocapture`

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::float_cmp,
        clippy::print_stdout
    )
)]
#![cfg(feature = "highs")]

use cobre_solver::{HighsSolver, RowBatch, SolverInterface, StageTemplate};

#[test]
fn q1_sign_convention_row_equality_vs_column_bound() {
    // ── LP-R: equality-row variant ─────────────────────────────────────────
    // 3 columns, 2 rows: row 0 = x2+x3 >= 4, row 1 = x1 == 7.
    let template_r = StageTemplate {
        num_cols: 3,
        num_rows: 2,
        num_nz: 3,
        // CSC: col 0 has 1 entry (row 1), col 1 has 1 entry (row 0),
        // col 2 has 1 entry (row 0).
        col_starts: vec![0_i32, 1, 2, 3],
        row_indices: vec![1_i32, 0, 0],
        values: vec![1.0, 1.0, 1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![10.0, 10.0, 10.0],
        objective: vec![5.0, 2.0, 3.0],
        row_lower: vec![4.0, 7.0],
        row_upper: vec![f64::INFINITY, 7.0],
        n_state: 0,
        n_transfer: 0,
        n_dual_relevant: 0,
        n_hydro: 0,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };
    let empty_rows = RowBatch {
        num_rows: 0,
        row_starts: vec![0_i32],
        col_indices: Vec::new(),
        values: Vec::new(),
        row_lower: Vec::new(),
        row_upper: Vec::new(),
    };
    let mut solver_r = HighsSolver::new().expect("HiGHS init");
    solver_r.load_model(&template_r);
    solver_r.add_rows(&empty_rows);
    let view_r = solver_r.solve(None).expect("LP-R must solve");
    let x_r = view_r.primal.to_vec();
    let dual_r = view_r.dual.to_vec();
    let rc_r = view_r.reduced_costs.to_vec();
    let obj_r = view_r.objective;
    // view_r is Copy; let the borrow end naturally by not using it further.

    // ── LP-C: column-bound variant ─────────────────────────────────────────
    // 3 columns, 1 row: row 0 = x2+x3 >= 4. x1 is fixed via col bounds.
    let template_c = StageTemplate {
        num_cols: 3,
        num_rows: 1,
        num_nz: 2,
        col_starts: vec![0_i32, 0, 1, 2],
        row_indices: vec![0_i32, 0],
        values: vec![1.0, 1.0],
        col_lower: vec![7.0, 0.0, 0.0],
        col_upper: vec![7.0, 10.0, 10.0],
        objective: vec![5.0, 2.0, 3.0],
        row_lower: vec![4.0],
        row_upper: vec![f64::INFINITY],
        n_state: 0,
        n_transfer: 0,
        n_dual_relevant: 0,
        n_hydro: 0,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };
    let mut solver_c = HighsSolver::new().expect("HiGHS init");
    solver_c.load_model(&template_c);
    solver_c.add_rows(&empty_rows);
    let view_c = solver_c.solve(None).expect("LP-C must solve");
    let x_c = view_c.primal.to_vec();
    let dual_c = view_c.dual.to_vec();
    let rc_c = view_c.reduced_costs.to_vec();
    let obj_c = view_c.objective;
    // view_c is Copy; let the borrow end naturally by not using it further.

    println!("\n=== LP-R (equality-row variant) ===");
    println!("  primal x        = {x_r:?}");
    println!("  dual (rows)     = {dual_r:?}   <- expect [y_capacity, y_fixing]");
    println!("  reduced_costs   = {rc_r:?}");
    println!("  objective       = {obj_r}");
    println!("\n=== LP-C (column-bound variant) ===");
    println!("  primal x        = {x_c:?}");
    println!("  dual (rows)     = {dual_c:?}   <- expect [y_capacity]");
    println!("  reduced_costs   = {rc_c:?}      <- focus on reduced_cost[0]");
    println!("  objective       = {obj_c}");

    println!("\n=== Comparison: cut-subgradient candidates ===");
    println!("  LP-R: dual[fixing_row] (row 1)     = {}", dual_r[1]);
    println!("  LP-C: reduced_cost[x1]   (col 0)   = {}", rc_c[0]);
    println!(
        "  Same value (numerically)?           {}",
        (dual_r[1] - rc_c[0]).abs() < 1e-8
    );
    println!(
        "  Same sign?                          {}",
        dual_r[1].signum() == rc_c[0].signum()
    );
    println!(
        "  Equality-row dual sign convention   {}",
        if dual_r[1] > 0.0 { "+" } else { "-" }
    );
    println!(
        "  Column reduced cost sign convention {}",
        if rc_c[0] > 0.0 { "+" } else { "-" }
    );

    // Both LPs should produce the same optimum.
    assert!(
        (obj_r - obj_c).abs() < 1e-8,
        "objectives must match: {obj_r} vs {obj_c}"
    );
    assert!(
        (x_r[0] - x_c[0]).abs() < 1e-8,
        "x1 must match: {} vs {}",
        x_r[0],
        x_c[0]
    );
    assert!(
        (x_r[1] - x_c[1]).abs() < 1e-8,
        "x2 must match: {} vs {}",
        x_r[1],
        x_c[1]
    );
    assert!(
        (x_r[2] - x_c[2]).abs() < 1e-8,
        "x3 must match: {} vs {}",
        x_r[2],
        x_c[2]
    );
}
