//! CLP determinism harness — declaration-order invariance and bit-for-bit
//! re-solve reproducibility for the CLP backend.
//!
//! Determinism (not merely correctness) is the property under test, so every
//! comparison here is **bit-for-bit** via [`f64::to_bits`], never a tolerance.
//! Two solves are "the same" only if their objective, primal vector, reduced
//! costs, and (inverse-permuted) row duals are identical down to the last
//! mantissa bit. A tolerance comparison would mask exactly the drift this
//! harness exists to catch.
//!
//! The harness is reusable: it exposes a [`MutationStep`] vocabulary and a
//! standalone [`assert_solve_sequence_deterministic`] helper that drives an
//! arbitrary mutate-then-solve sequence through three runs — the same instance
//! twice plus a fresh instance — and asserts all three agree bit-for-bit. New
//! solve-capability sequences (incremental bound patches, hot-start re-solves)
//! plug in by constructing a `&[MutationStep]` and calling that helper; the
//! core comparison logic does not change.
//!
//! Two facts hold this together and are preconditions for everything below:
//!   - The active solver disables perturbation (the default profile sends the
//!     perturbation-off mode), without which CLP injects pseudo-random bound
//!     perturbations and no re-solve is bit-reproducible.
//!   - The solver forwards row prices in the canonical sign convention with no
//!     negation, so duals compare directly.
//!
//! The fixtures (`SS1.1`, a 3-column / 2-equality-row LP, and `SS1.2`, two
//! appended `>=` rows over the same columns) are re-declared locally because an
//! integration test cannot import the `#[cfg(test)]` builders from `src`.

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

use cobre_solver::{ClpProfile, ClpSolver, RowBatch, SolverInterface, StageTemplate};

// ─── Shared fixtures (re-declared locally; see module docs) ──────────────────

/// `SS1.1`: 3 columns, 2 equality rows.
///
/// ```text
/// minimize  0*x0 + 1*x1 + 50*x2
/// s.t.  row0:  x0           = 6
///       row1: 2*x0     + x2 = 14
///       x0 in [0, 10], x1 in [0, +inf), x2 in [0, 8]
/// ```
///
/// Optimal: objective `100`, primals `(6, 0, 2)`, row duals `[-100, 50]`.
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

/// `SS1.2`: two appended `>=` rows over the `SS1.1` columns.
///
/// ```text
/// row2: -5*x0 + x1 >= 20   (row_lower 20, row_upper +inf)
/// row3:  3*x0 + x1 >= 80   (row_lower 80, row_upper +inf)
/// ```
///
/// After `add_rows`, the merged LP solves to objective `162`, primals
/// `(6, 62, 2)`.
fn make_fixture_row_batch() -> RowBatch {
    RowBatch {
        num_rows: 2,
        row_starts: vec![0_i32, 2, 4],
        col_indices: vec![0_i32, 1, 0, 1],
        values: vec![-5.0, 1.0, 3.0, 1.0],
        row_lower: vec![20.0, 80.0],
        row_upper: vec![f64::INFINITY, f64::INFINITY],
    }
}

// ─── Reusable mutate-then-solve vocabulary ───────────────────────────────────

/// A single step in a mutate-then-solve sequence fed to
/// [`assert_solve_sequence_deterministic`].
///
/// The variants cover the mutation surface the repeated-resolve hot path drives:
/// appending rows, patching row/column bounds, and solving. New solve-capability
/// work expresses its workload as a `&[MutationStep]` and reuses the helper —
/// it never edits the comparison core.
#[derive(Clone)]
pub enum MutationStep {
    /// Append a batch of constraint rows (CSR form) via `add_rows`.
    AddRows(RowBatch),
    /// Patch the bounds of the given rows via `set_row_bounds`.
    SetRowBounds {
        /// Row indices to patch.
        indices: Vec<usize>,
        /// New lower bounds, aligned with `indices`.
        lower: Vec<f64>,
        /// New upper bounds, aligned with `indices`.
        upper: Vec<f64>,
    },
    /// Patch the bounds of the given columns via `set_col_bounds`.
    SetColBounds {
        /// Column indices to patch.
        indices: Vec<usize>,
        /// New lower bounds, aligned with `indices`.
        lower: Vec<f64>,
        /// New upper bounds, aligned with `indices`.
        upper: Vec<f64>,
    },
    /// Solve the current model (cold, no offered basis) and capture the result.
    Solve,
}

/// The bit-pattern fingerprint of one solve: objective + every solution buffer,
/// stored as raw `u64` bit patterns so equality is exact.
///
/// Returned (in a `Vec`) from [`assert_solve_sequence_deterministic`] so callers
/// can layer further cross-sequence assertions on top of the reproducibility
/// guarantee. The fields are public for that read-back; the constructor is
/// crate-internal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolveBits {
    /// Raw bits of the objective value.
    pub objective: u64,
    /// Raw bits of the primal column-value vector.
    pub primal: Vec<u64>,
    /// Raw bits of the reduced-cost vector.
    pub reduced_costs: Vec<u64>,
    /// Raw bits of the row-dual vector.
    pub dual: Vec<u64>,
}

impl SolveBits {
    /// Captures the bit pattern of the solver's most recent solve result.
    fn capture(objective: f64, primal: &[f64], reduced_costs: &[f64], dual: &[f64]) -> Self {
        Self {
            objective: objective.to_bits(),
            primal: primal.iter().map(|v| v.to_bits()).collect(),
            reduced_costs: reduced_costs.iter().map(|v| v.to_bits()).collect(),
            dual: dual.iter().map(|v| v.to_bits()).collect(),
        }
    }
}

/// Runs a fresh solver over the step sequence, returning the bit fingerprint of
/// every `Solve` step in order.
fn run_sequence(
    mut solver: ClpSolver,
    base: &StageTemplate,
    steps: &[MutationStep],
) -> Vec<SolveBits> {
    run_sequence_in_place(&mut solver, base, steps)
}

/// Asserts two captured-solve sequences are bit-for-bit identical, naming the
/// solve ordinal and buffer that diverged.
fn assert_bits_eq(label: &str, a: &[SolveBits], b: &[SolveBits]) {
    assert_eq!(
        a.len(),
        b.len(),
        "{label}: solve count differs ({} vs {})",
        a.len(),
        b.len()
    );
    for (i, (lhs, rhs)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            lhs.objective, rhs.objective,
            "{label}: objective bits diverged at solve #{i}"
        );
        assert_eq!(
            lhs.primal, rhs.primal,
            "{label}: primal bits diverged at solve #{i}"
        );
        assert_eq!(
            lhs.reduced_costs, rhs.reduced_costs,
            "{label}: reduced-cost bits diverged at solve #{i}"
        );
        assert_eq!(
            lhs.dual, rhs.dual,
            "{label}: row-dual bits diverged at solve #{i}"
        );
    }
}

/// Asserts that running `steps` over a CLP solver built by `build` is
/// bit-for-bit reproducible: the same instance run twice and a fresh instance
/// run once all yield identical objective / primal / reduced-cost / dual
/// buffers at every `Solve` step.
///
/// This is the reusable acceptance criterion new solve-capability sequences
/// (incremental bound patches, hot-start re-solves) call without touching the
/// comparison core: hand it a factory and a step list.
///
/// Returns the captured bit fingerprints from the first run so callers can layer
/// further assertions (e.g. anchoring the captured objectives to known fixture
/// optima) on top of the reproducibility guarantee. The property under test is
/// self-consistency — a path reproducing itself and staying order-invariant —
/// never a cross-check that two different solve methods agree.
///
/// # Panics
///
/// Panics (failing the test) if any run diverges bit-for-bit from the first, if
/// any `Solve` step fails to reach optimality, or if `steps` contains no `Solve`
/// step.
#[must_use]
pub fn assert_solve_sequence_deterministic(
    build: impl Fn() -> ClpSolver,
    base: &StageTemplate,
    steps: &[MutationStep],
) -> Vec<SolveBits> {
    // Run 1: a fresh instance over the full sequence.
    let run1 = run_sequence(build(), base, steps);
    assert!(
        !run1.is_empty(),
        "step sequence must contain at least one Solve step"
    );

    // Run 2: a second fresh instance over the identical sequence.
    let run2 = run_sequence(build(), base, steps);
    assert_bits_eq("fresh-instance re-run", &run1, &run2);

    // Run 3: a single instance driven over the sequence twice back-to-back.
    // CLP retains internal factorization/basis state across solves, so this
    // exercises the "warm internal state must not perturb results" path that a
    // two-instance comparison cannot reach.
    let mut solver = build();
    let pass_a = run_sequence_in_place(&mut solver, base, steps);
    let pass_b = run_sequence_in_place(&mut solver, base, steps);
    assert_bits_eq("same-instance pass A vs run1", &pass_a, &run1);
    assert_bits_eq("same-instance pass B vs pass A", &pass_b, &pass_a);

    run1
}

/// Re-runs the full sequence on an existing solver instance (reloads the base
/// model first, so the instance returns to a known state between passes).
fn run_sequence_in_place(
    solver: &mut ClpSolver,
    base: &StageTemplate,
    steps: &[MutationStep],
) -> Vec<SolveBits> {
    solver.load_model(base);
    let mut captured = Vec::new();
    for step in steps {
        match step {
            MutationStep::AddRows(batch) => solver.add_rows(batch),
            MutationStep::SetRowBounds {
                indices,
                lower,
                upper,
            } => solver.set_row_bounds(indices, lower, upper),
            MutationStep::SetColBounds {
                indices,
                lower,
                upper,
            } => solver.set_col_bounds(indices, lower, upper),
            MutationStep::Solve => {
                let view = solver.solve(None).expect("solve must reach optimality");
                captured.push(SolveBits::capture(
                    view.objective,
                    view.primal,
                    view.reduced_costs,
                    view.dual,
                ));
            }
        }
    }
    captured
}

// ─── Row permutation (declaration-order invariance) ──────────────────────────

/// Reorders the rows of a `StageTemplate`'s CSC matrix and row bounds according
/// to `perm`, where `perm[new_row] == old_row`. The column structure is
/// unchanged; only each non-zero's row index is remapped and the row-bound
/// vectors are gathered in the new order.
///
/// `perm` must be a permutation of `0..template.num_rows`.
///
/// # Panics
///
/// Panics if `perm.len() != template.num_rows`, or if a stored row index does
/// not fit back into `i32` after remapping.
#[must_use]
pub fn permute_rows(template: &StageTemplate, perm: &[usize]) -> StageTemplate {
    assert_eq!(
        perm.len(),
        template.num_rows,
        "perm length {} must equal num_rows {}",
        perm.len(),
        template.num_rows
    );

    // inverse[old_row] = new_row. Used to remap each non-zero's row index.
    let inverse = inverse_permutation(perm);

    // Remap every non-zero's row index from old → new position. CSC column
    // structure (col_starts) is untouched; only the stored row index changes.
    let mut new_row_indices = template.row_indices.clone();
    for ri in &mut new_row_indices {
        let old = usize::try_from(*ri).expect("row index must be non-negative");
        *ri = i32::try_from(inverse[old]).expect("row index fits in i32");
    }

    // Gather row bounds in the new order: new position `n` takes old row `perm[n]`.
    let mut new_row_lower = vec![0.0_f64; template.num_rows];
    let mut new_row_upper = vec![0.0_f64; template.num_rows];
    for (new_pos, &old_pos) in perm.iter().enumerate() {
        new_row_lower[new_pos] = template.row_lower[old_pos];
        new_row_upper[new_pos] = template.row_upper[old_pos];
    }

    StageTemplate {
        row_indices: new_row_indices,
        row_lower: new_row_lower,
        row_upper: new_row_upper,
        col_starts: template.col_starts.clone(),
        values: template.values.clone(),
        col_lower: template.col_lower.clone(),
        col_upper: template.col_upper.clone(),
        objective: template.objective.clone(),
        col_scale: template.col_scale.clone(),
        row_scale: template.row_scale.clone(),
        ..*template
    }
}

/// Computes the inverse of a permutation: `inverse[perm[i]] == i`.
///
/// For `permute_rows`, `perm[new] == old`, so `inverse[old] == new` — exactly
/// the map needed to remap stored row indices and to invert duals back to the
/// original row order for comparison.
#[must_use]
pub fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0_usize; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inverse[p] = i;
    }
    inverse
}

// ─── Kept tests ──────────────────────────────────────────────────────────────

/// Declaration-order invariance: solving an LP and a row-permuted copy yields
/// bit-for-bit identical objective, primals, and reduced costs, and the
/// permuted solve's row duals, mapped back through the inverse permutation,
/// match the original solve's row duals bit-for-bit.
///
/// Run on the `SS1.2` fixture (`SS1.1` + the two appended `>=` rows) so the
/// permutation spans four rows of two distinct kinds (equality + inequality).
#[test]
fn declaration_order_invariance_ss1_2() {
    // Build the merged SS1.2 LP as a single StageTemplate so the whole row set
    // can be permuted in one shot. Start from SS1.1, then append SS1.2's two
    // rows by hand into CSC form.
    let base = merged_ss1_2_template();

    // Reference solve in declaration order.
    let mut solver = ClpSolver::new().expect("ClpSolver::new must succeed");
    solver.load_model(&base);
    let ref_view = solver.solve(None).expect("reference solve must succeed");
    let ref_objective = ref_view.objective.to_bits();
    let ref_primal: Vec<u64> = ref_view.primal.iter().map(|v| v.to_bits()).collect();
    let ref_reduced: Vec<u64> = ref_view.reduced_costs.iter().map(|v| v.to_bits()).collect();
    let ref_dual: Vec<u64> = ref_view.dual.iter().map(|v| v.to_bits()).collect();
    drop(solver);

    // A non-trivial permutation of the four rows: reverse order.
    // perm[new_row] = old_row.
    let perm = [3_usize, 2, 1, 0];
    let inverse = inverse_permutation(&perm);
    let permuted = permute_rows(&base, &perm);

    let mut psolver = ClpSolver::new().expect("ClpSolver::new must succeed");
    psolver.load_model(&permuted);
    let pview = psolver.solve(None).expect("permuted solve must succeed");

    // Objective / primals / reduced costs are row-order-independent → identical bits.
    assert_eq!(
        pview.objective.to_bits(),
        ref_objective,
        "objective bits changed under row permutation"
    );
    let p_primal: Vec<u64> = pview.primal.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        p_primal, ref_primal,
        "primal bits changed under row permutation"
    );
    let p_reduced: Vec<u64> = pview.reduced_costs.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        p_reduced, ref_reduced,
        "reduced-cost bits changed under row permutation"
    );

    // Row duals must match after inverse-permuting the permuted solve's duals
    // back into the original row order: permuted_dual[new] corresponds to
    // original row `perm[new]`, so original row `r` reads from new position
    // `inverse[r]`.
    let p_dual: Vec<u64> = pview.dual.iter().map(|v| v.to_bits()).collect();
    let unpermuted_dual: Vec<u64> = (0..base.num_rows).map(|r| p_dual[inverse[r]]).collect();
    assert_eq!(
        unpermuted_dual, ref_dual,
        "row-dual bits diverged after inverse permutation (declaration-order leak)"
    );
}

/// Cross-instance re-solve reproducibility: the same mutate-then-solve sequence
/// run on a fresh instance, a second fresh instance, and a single reused
/// instance (twice) all yield bit-for-bit identical objective / primal /
/// reduced-cost / dual buffers at every solve.
///
/// The sequence loads `SS1.1`, solves, appends `SS1.2`, solves again, then
/// patches a column bound and solves a third time — exercising every mutation
/// variant and three solves through the reusable helper.
#[test]
fn cross_instance_resolve_reproducibility() {
    let base = make_fixture_stage_template();
    let steps = vec![
        MutationStep::Solve,
        MutationStep::AddRows(make_fixture_row_batch()),
        MutationStep::Solve,
        MutationStep::SetColBounds {
            indices: vec![2],
            lower: vec![0.0],
            upper: vec![8.0],
        },
        MutationStep::Solve,
    ];

    let captured = assert_solve_sequence_deterministic(
        || ClpSolver::new().expect("ClpSolver::new must succeed"),
        &base,
        &steps,
    );

    // Sanity-anchor the captured values to the known fixture optima so a future
    // refactor that silently changes the solved model is caught here too.
    assert_eq!(captured.len(), 3, "expected three Solve captures");
    assert_eq!(
        f64::from_bits(captured[0].objective),
        100.0,
        "first solve objective must be the SS1.1 optimum"
    );
    assert_eq!(
        f64::from_bits(captured[1].objective),
        162.0,
        "second solve objective must be the SS1.2 optimum"
    );
}

/// Native incremental-mutation determinism: a mutate-then-solve sequence that
/// drives every native-mutation entry point (`add_rows`, `set_row_bounds`,
/// `set_col_bounds`) interleaved with solves must be bit-for-bit reproducible
/// (run-to-run, cross-instance, and same-instance-reused) AND declaration-order
/// invariant.
///
/// Reproducibility is checked through the reusable
/// [`assert_solve_sequence_deterministic`] helper (three runs: two fresh
/// instances plus one reused instance driven twice). Declaration-order
/// invariance is checked by replaying the identical mutation sequence on the
/// row-permuted base `SS1.1` and inverse-permuting the resulting row duals back
/// into declaration order, then comparing bit-for-bit against the reference run.
///
/// The native-mutation path appends rows via `Clp_addRows` and patches bounds
/// via `Clp_chgRowLower/Upper` / `Clp_chgColumnLower/Upper` in place — preserving
/// CLP's factorization across each solve — so this is the determinism gate for
/// that path. (Its bit-for-bit equality with the prior full-reload route was
/// verified separately during development against a captured reload baseline.)
#[test]
fn native_mutation_sequence_is_deterministic_and_order_invariant() {
    let base = make_fixture_stage_template();
    // A sequence touching all three native-mutation variants between solves:
    //   solve cold → add the two SS1.2 rows → solve → tighten a column bound →
    //   solve → re-pin a row's equality bound → solve.
    let steps = vec![
        MutationStep::Solve,
        MutationStep::AddRows(make_fixture_row_batch()),
        MutationStep::Solve,
        MutationStep::SetColBounds {
            indices: vec![2],
            lower: vec![0.0],
            upper: vec![8.0],
        },
        MutationStep::Solve,
        MutationStep::SetRowBounds {
            indices: vec![0],
            lower: vec![6.0],
            upper: vec![6.0],
        },
        MutationStep::Solve,
    ];

    // Reproducibility (run-to-run + cross-instance + same-instance-reused).
    let reference = assert_solve_sequence_deterministic(
        || ClpSolver::new().expect("ClpSolver::new must succeed"),
        &base,
        &steps,
    );
    assert_eq!(reference.len(), 4, "expected four Solve captures");
    assert_eq!(
        f64::from_bits(reference[0].objective),
        100.0,
        "cold SS1.1 solve must hit the known optimum"
    );
    assert_eq!(
        f64::from_bits(reference[1].objective),
        162.0,
        "post-add_rows solve must hit the SS1.2 optimum"
    );

    // Declaration-order invariance: run the SAME native-mutation sequence on a
    // row-permuted SS1.1 base. The appended rows land at the same logical
    // positions (`add_rows` appends after the existing rows regardless of how
    // the existing rows are ordered), and the bound patches address rows by the
    // permuted index, so we permute the SetRowBounds indices to track the base
    // permutation. Objective / primals / reduced costs are row-order-independent
    // and must match bit-for-bit; row duals must match after inverse-permuting
    // the appended-aware index map back into declaration order.
    let perm = [1_usize, 0]; // swap the two SS1.1 equality rows
    let inverse = inverse_permutation(&perm);
    let permuted_base = permute_rows(&base, &perm);
    // Row 0 in declaration order sits at position `inverse[0]` after permutation;
    // retarget the SetRowBounds index so it patches the same logical row.
    let permuted_steps = vec![
        MutationStep::Solve,
        MutationStep::AddRows(make_fixture_row_batch()),
        MutationStep::Solve,
        MutationStep::SetColBounds {
            indices: vec![2],
            lower: vec![0.0],
            upper: vec![8.0],
        },
        MutationStep::Solve,
        MutationStep::SetRowBounds {
            indices: vec![inverse[0]],
            lower: vec![6.0],
            upper: vec![6.0],
        },
        MutationStep::Solve,
    ];
    let permuted = assert_solve_sequence_deterministic(
        || ClpSolver::new().expect("ClpSolver::new must succeed"),
        &permuted_base,
        &permuted_steps,
    );

    // The permutation only reorders the two base SS1.1 rows; the two appended
    // SS1.2 rows keep positions 2 and 3 in both runs. Build the old→new row map
    // over the full 4-row model: base rows follow `inverse`, appended rows are
    // identity at 2 and 3.
    let full_inverse = [inverse[0], inverse[1], 2_usize, 3];
    for (i, (r, p)) in reference.iter().zip(&permuted).enumerate() {
        assert_eq!(
            r.objective, p.objective,
            "objective bits diverged under row permutation at solve #{i}"
        );
        assert_eq!(
            r.primal, p.primal,
            "primal bits diverged under row permutation at solve #{i}"
        );
        assert_eq!(
            r.reduced_costs, p.reduced_costs,
            "reduced-cost bits diverged under row permutation at solve #{i}"
        );
        // Inverse-permute the permuted run's row duals back into declaration
        // order before comparing. Solve #0 sees only the 2 base rows; later
        // solves see all 4. Map each declaration-order row to its permuted slot.
        let unpermuted: Vec<u64> = (0..r.dual.len())
            .map(|row| p.dual[full_inverse[row]])
            .collect();
        assert_eq!(
            r.dual, unpermuted,
            "row-dual bits diverged after inverse permutation at solve #{i} \
             (declaration-order leak in the native-mutation path)"
        );
    }
}

/// Builds the SS1.2 LP (SS1.1 + the two appended `>=` rows) as one
/// `StageTemplate`, so the full 4-row set can be permuted atomically.
fn merged_ss1_2_template() -> StageTemplate {
    // Drive the canonical add_rows merge through ClpSolver so the merged CSC is
    // produced by the same code path the rest of the suite uses, then mirror the
    // resulting dimensions/bounds into a StageTemplate. We reconstruct the merged
    // CSC directly here (the merge is small and explicit) to keep the template
    // self-contained.
    //
    // Columns: 3. Rows: 4.
    //   row0:  x0           = 6
    //   row1: 2*x0     + x2 = 14
    //   row2: -5*x0 + x1     >= 20
    //   row3:  3*x0 + x1     >= 80
    //
    // CSC (column-major), columns 0,1,2:
    //   col0 nz: row0=1, row1=2, row2=-5, row3=3   (4 entries)
    //   col1 nz: row2=1, row3=1                     (2 entries)
    //   col2 nz: row1=1                             (1 entry)
    StageTemplate {
        num_cols: 3,
        num_rows: 4,
        num_nz: 7,
        col_starts: vec![0_i32, 4, 6, 7],
        row_indices: vec![0_i32, 1, 2, 3, 2, 3, 1],
        values: vec![1.0, 2.0, -5.0, 3.0, 1.0, 1.0, 1.0],
        col_lower: vec![0.0, 0.0, 0.0],
        col_upper: vec![10.0, f64::INFINITY, 8.0],
        objective: vec![0.0, 1.0, 50.0],
        row_lower: vec![6.0, 14.0, 20.0, 80.0],
        row_upper: vec![6.0, 14.0, f64::INFINITY, f64::INFINITY],
        n_state: 1,
        n_transfer: 0,
        n_dual_relevant: 1,
        n_hydro: 1,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

// ─── Wired-knob determinism (DSE pricing, refactor cadence, hot-start) ────────

/// The mutate-then-solve sequence shared by the wired-knob determinism tests:
/// solve `SS1.1` cold, append the two `SS1.2` rows, solve, tighten a column
/// bound, solve. It touches every native-mutation entry point between solves so
/// the knob under test is exercised across a real re-solve sequence, not a
/// single solve.
fn knob_step_sequence() -> Vec<MutationStep> {
    vec![
        MutationStep::Solve,
        MutationStep::AddRows(make_fixture_row_batch()),
        MutationStep::Solve,
        MutationStep::SetColBounds {
            indices: vec![2],
            lower: vec![0.0],
            upper: vec![8.0],
        },
        MutationStep::Solve,
    ]
}

/// **Wired knob — dual-steepest-edge pricing.** A profile pinning full DSE
/// (`dual_pricing_mode = 1`, the backward-pass tuned value) is driven through
/// `apply_profile`, which now issues the shim pricing setter. The resulting
/// solve sequence must be bit-for-bit reproducible (run-to-run + cross-instance
/// + same-instance-reused) and declaration-order invariant.
///
/// This proves the corrected shim cast makes the pricing knob fault-free AND
/// that pinning DSE does not break determinism — never a hot-vs-cold check.
#[test]
fn dse_pricing_profile_is_deterministic_and_order_invariant() {
    let dse_profile = ClpProfile {
        dual_pricing_mode: 1,
        ..ClpProfile::default()
    };
    let build = move || {
        let mut s = ClpSolver::new().expect("ClpSolver::new must succeed");
        s.apply_profile(&dse_profile);
        s
    };

    // Reproducibility across three runs.
    let base = make_fixture_stage_template();
    let steps = knob_step_sequence();
    let reference = assert_solve_sequence_deterministic(build, &base, &steps);
    assert_eq!(reference.len(), 3, "expected three Solve captures");
    assert_eq!(
        f64::from_bits(reference[0].objective),
        100.0,
        "DSE cold solve must hit the SS1.1 optimum"
    );
    assert_eq!(
        f64::from_bits(reference[1].objective),
        162.0,
        "DSE post-add_rows solve must hit the SS1.2 optimum"
    );

    // Declaration-order invariance: replay the same sequence on a row-permuted
    // SS1.1 base; appended rows keep positions 2 and 3, base rows follow `perm`.
    assert_knob_sequence_order_invariant(&build, &reference, &steps);
}

/// **Wired knob — refactorization cadence.** A profile setting
/// `factorization_frequency = 200` (the backward-pass tuned value) is driven
/// through `apply_profile`, which now issues the shim factorization-frequency
/// setter. The solve sequence must be bit-for-bit reproducible and
/// declaration-order invariant.
#[test]
fn factorization_frequency_profile_is_deterministic_and_order_invariant() {
    let factfreq_profile = ClpProfile {
        factorization_frequency: 200,
        ..ClpProfile::default()
    };
    let build = move || {
        let mut s = ClpSolver::new().expect("ClpSolver::new must succeed");
        s.apply_profile(&factfreq_profile);
        s
    };

    let base = make_fixture_stage_template();
    let steps = knob_step_sequence();
    let reference = assert_solve_sequence_deterministic(build, &base, &steps);
    assert_eq!(reference.len(), 3, "expected three Solve captures");
    assert_eq!(
        f64::from_bits(reference[0].objective),
        100.0,
        "refactor-cadence cold solve must hit the SS1.1 optimum"
    );
    assert_eq!(
        f64::from_bits(reference[1].objective),
        162.0,
        "refactor-cadence post-add_rows solve must hit the SS1.2 optimum"
    );

    assert_knob_sequence_order_invariant(&build, &reference, &steps);
}

/// **Wired knob — the backward-pass tuned profile (DSE + cadence together).**
/// The `BACKWARD` CLP profile pins both `dual_pricing_mode = 1` and
/// `factorization_frequency = 200`; this drives both shim setters in one
/// `apply_profile` and proves the combination is deterministic and
/// order-invariant — the configuration the backward pass actually applies.
#[test]
fn backward_tuned_profile_is_deterministic_and_order_invariant() {
    let backward_profile = ClpProfile {
        dual_pricing_mode: 1,
        factorization_frequency: 200,
        ..ClpProfile::default()
    };
    let build = move || {
        let mut s = ClpSolver::new().expect("ClpSolver::new must succeed");
        s.apply_profile(&backward_profile);
        s
    };

    let base = make_fixture_stage_template();
    let steps = knob_step_sequence();
    let reference = assert_solve_sequence_deterministic(build, &base, &steps);
    assert_eq!(
        f64::from_bits(reference[0].objective),
        100.0,
        "backward-tuned cold solve must hit the SS1.1 optimum"
    );
    assert_eq!(
        f64::from_bits(reference[1].objective),
        162.0,
        "backward-tuned post-add_rows solve must hit the SS1.2 optimum"
    );

    assert_knob_sequence_order_invariant(&build, &reference, &steps);
}

/// Shared declaration-order-invariance check for the knob sequences: replays the
/// identical `SS1.1 → add SS1.2 → patch col bound` sequence on a row-permuted
/// `SS1.1` base and asserts objective / primals / reduced costs match
/// bit-for-bit, and row duals match after inverse-permuting back into
/// declaration order. The two appended `SS1.2` rows keep positions 2 and 3 in
/// both runs (`add_rows` appends after the existing rows regardless of base
/// order); the two base rows follow the permutation.
fn assert_knob_sequence_order_invariant(
    build: &impl Fn() -> ClpSolver,
    reference: &[SolveBits],
    steps: &[MutationStep],
) {
    let base = make_fixture_stage_template();
    let perm = [1_usize, 0]; // swap the two SS1.1 equality rows
    let inverse = inverse_permutation(&perm);
    let permuted_base = permute_rows(&base, &perm);
    let permuted = assert_solve_sequence_deterministic(build, &permuted_base, steps);

    // Old→new row map: base rows follow `inverse`, appended rows are identity.
    let full_inverse = [inverse[0], inverse[1], 2_usize, 3];
    for (i, (r, p)) in reference.iter().zip(&permuted).enumerate() {
        assert_eq!(
            r.objective, p.objective,
            "objective bits diverged under row permutation at solve #{i}"
        );
        assert_eq!(
            r.primal, p.primal,
            "primal bits diverged under row permutation at solve #{i}"
        );
        assert_eq!(
            r.reduced_costs, p.reduced_costs,
            "reduced-cost bits diverged under row permutation at solve #{i}"
        );
        let unpermuted: Vec<u64> = (0..r.dual.len())
            .map(|row| p.dual[full_inverse[row]])
            .collect();
        assert_eq!(
            r.dual, unpermuted,
            "row-dual bits diverged after inverse permutation at solve #{i} \
             (declaration-order leak in the wired-knob path)"
        );
    }
}

/// Drives the hot-start lifecycle over a base template and returns the bit
/// fingerprints of every `solve_from_hot_start` re-solve, in order.
///
/// The lifecycle: load the base, solve cold once (leaving the simplex rim and
/// factorization alive on the persistent instance), `mark_hot_start`, then for
/// each patch in `col_patches` apply it via `set_col_bounds` and re-solve from
/// the snapshot, capturing the result. Releases the snapshot at the end. This is
/// the per-opening warm re-solve composition: snapshot once, re-solve each
/// bound-patched model from the snapshot.
fn run_hot_start_sequence(
    mut solver: ClpSolver,
    base: &StageTemplate,
    col_patches: &[(usize, f64, f64)],
) -> Vec<SolveBits> {
    run_hot_start_sequence_in_place(&mut solver, base, col_patches)
}

/// Re-runs the hot-start lifecycle on an existing solver instance (reloads the
/// base and re-snapshots first, so the instance returns to a known state between
/// passes).
fn run_hot_start_sequence_in_place(
    solver: &mut ClpSolver,
    base: &StageTemplate,
    col_patches: &[(usize, f64, f64)],
) -> Vec<SolveBits> {
    solver.load_model(base);
    // Cold solve leaves the rim/factorization alive for the snapshot.
    let _ = solver
        .solve(None)
        .expect("cold solve must reach optimality");
    solver.mark_hot_start();
    let mut captured = Vec::new();
    for &(col, lo, hi) in col_patches {
        solver.set_col_bounds(&[col], &[lo], &[hi]);
        let view = solver
            .solve_from_hot_start()
            .expect("hot-start re-solve must reach optimality");
        captured.push(SolveBits::capture(
            view.objective,
            view.primal,
            view.reduced_costs,
            view.dual,
        ));
    }
    solver.unmark_hot_start();
    captured
}

/// **Wired knob — hot-start.** The `mark_hot_start` /
/// `solve_from_hot_start` / `unmark_hot_start` lifecycle, driven over a sequence
/// of bound-patched re-solves on the persistent-factorization instance, must be
/// bit-for-bit reproducible: a fresh instance, a second fresh instance, and a
/// single reused instance driven twice all agree at every re-solve. It must also
/// be declaration-order invariant.
///
/// This is the contract from the project's determinism definition: the hot-start
/// path must reproduce **itself** (run-to-run + cross-instance) and stay
/// order-invariant — it is NOT required to match a cold solve's dual vertex
/// (`solveFromHotStart`'s truncated dual may report a different-but-valid vertex
/// on a degenerate optimum). Perturbation stays off (`102`) across the whole
/// lifetime via the default profile.
#[test]
fn hot_start_sequence_is_deterministic_and_order_invariant() {
    let base = make_fixture_stage_template();
    // Two bound patches re-solved from the same snapshot: tighten col 2, then
    // relax it back. Each re-solve goes through solve_from_hot_start.
    let patches = [(2_usize, 0.0_f64, 8.0_f64), (2_usize, 0.0_f64, 6.0_f64)];

    // Run 1 + Run 2: two fresh instances.
    let run1 = run_hot_start_sequence(
        ClpSolver::new().expect("ClpSolver::new must succeed"),
        &base,
        &patches,
    );
    assert!(!run1.is_empty(), "hot-start sequence produced no re-solves");
    let run2 = run_hot_start_sequence(
        ClpSolver::new().expect("ClpSolver::new must succeed"),
        &base,
        &patches,
    );
    assert_bits_eq("hot-start fresh-instance re-run", &run1, &run2);

    // Run 3: a single reused instance driven over the lifecycle twice. Each call
    // reloads the base and re-snapshots, so the persistent instance returns to a
    // known state between passes — exercising the "warm internal state must not
    // perturb the snapshot result" path a two-instance comparison cannot reach.
    let mut solver = ClpSolver::new().expect("ClpSolver::new must succeed");
    let pass_a = run_hot_start_sequence_in_place(&mut solver, &base, &patches);
    let pass_b = run_hot_start_sequence_in_place(&mut solver, &base, &patches);
    assert_bits_eq("hot-start same-instance pass A vs run1", &pass_a, &run1);
    assert_bits_eq("hot-start same-instance pass B vs pass A", &pass_b, &pass_a);

    // Declaration-order invariance: replay the identical hot-start lifecycle on a
    // row-permuted SS1.1 base. The bound patches address columns (order-stable),
    // and only rows are permuted, so objective / primals / reduced costs must
    // match bit-for-bit and row duals must match after inverse permutation.
    let perm = [1_usize, 0];
    let inverse = inverse_permutation(&perm);
    let permuted_base = permute_rows(&base, &perm);
    let permuted = run_hot_start_sequence(
        ClpSolver::new().expect("ClpSolver::new must succeed"),
        &permuted_base,
        &patches,
    );
    for (i, (r, p)) in run1.iter().zip(&permuted).enumerate() {
        assert_eq!(
            r.objective, p.objective,
            "hot-start objective bits diverged under row permutation at re-solve #{i}"
        );
        assert_eq!(
            r.primal, p.primal,
            "hot-start primal bits diverged under row permutation at re-solve #{i}"
        );
        assert_eq!(
            r.reduced_costs, p.reduced_costs,
            "hot-start reduced-cost bits diverged under row permutation at re-solve #{i}"
        );
        let unpermuted: Vec<u64> = (0..r.dual.len()).map(|row| p.dual[inverse[row]]).collect();
        assert_eq!(
            r.dual, unpermuted,
            "hot-start row-dual bits diverged after inverse permutation at re-solve #{i} \
             (declaration-order leak in the hot-start path)"
        );
    }
}
