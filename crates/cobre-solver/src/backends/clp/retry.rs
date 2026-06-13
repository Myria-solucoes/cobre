//! Cold-solve escalation ladder for [`ClpSolver`](super::solver::ClpSolver).
//!
//! Additional `impl ClpSolver` block (the struct is owned by `solver`): the
//! `LADDER_RUNGS` ladder-size constant, the `EscalationOutcome` the ladder
//! returns by value, and the `escalate_run` / `escalate_solve` methods that
//! recover a feasible LP from CLP's spurious `PRIMAL_INFEASIBLE`. The ladder is
//! determinism-sensitive: fixed rung order, no randomness, no time-dependent
//! branching, so results stay bit-for-bit identical across thread / rank counts.

use std::time::Instant;

use super::config::ClpAlgorithm;
use super::solver::ClpSolver;
use crate::clp_ffi;

/// Number of re-solve rungs in [`ClpSolver::escalate_solve`].
///
/// The ladder runs (1) primal simplex, (2) perturbation-on dual then primal,
/// (3) scaling-on dual then primal — five re-solves total. `solve` charges this
/// many attempts to `retry_count` when the ladder is exhausted; the recovered
/// path charges only the rungs that actually ran (`EscalationOutcome::attempts`).
/// Typed `usize` so it can size the `RUNGS` array without a fallible cast; the
/// `retry_count` use-sites widen it to `u64`.
///
/// `pub(crate)` because the escalation-count contract is shared by the solve
/// path (`interface`'s `solve` charges this many attempts to `retry_count` on
/// exhaustion) and the regression test that asserts the exhausted-ladder
/// `retry_count`; it is re-exported from `clp/mod.rs` so both reach it.
pub(crate) const LADDER_RUNGS: usize = 5;

/// Outcome of a recovered [`ClpSolver::escalate_solve`] run.
///
/// Returned by value (not a borrow) so the caller can finish updating
/// `self.stats` before constructing the `SolutionView` that borrows the owned
/// solution buffers, keeping the aliasing discipline identical to the happy
/// path. The solution itself is already copied into `col_value` / `col_dual` /
/// `row_dual` by `escalate_solve` on the rung that returned OPTIMAL.
#[derive(Debug, Clone, Copy)]
pub(super) struct EscalationOutcome {
    /// Objective value of the recovered optimal solution (minimize sense).
    pub(super) objective: f64,
    /// Simplex iterations reported by the recovering rung.
    pub(super) iterations: u64,
    /// Wall-clock seconds spent across all rungs that ran.
    pub(super) solve_time: f64,
    /// Number of re-solve rungs that actually executed (1..=`LADDER_RUNGS`).
    pub(super) attempts: u64,
}

impl ClpSolver {
    /// Runs one escalation rung's re-solve from a clean cold basis.
    ///
    /// Resets the model to the all-slack cold basis (see
    /// [`Self::reset_cold_basis`]) and then runs the requested simplex variant.
    /// Returns the CLP solve status int (`0 = optimal`). The cold reset before
    /// every rung removes the inherited-bad-basis risk: each rung starts from a
    /// well-defined basis rather than the failed basis left by `Clp_dual`.
    fn escalate_run(&mut self, algorithm: ClpAlgorithm) -> i32 {
        self.reset_cold_basis();
        match algorithm {
            ClpAlgorithm::Dual => {
                // SAFETY: `self.handle` is a valid, non-null CLP pointer with a
                // model loaded (asserted via `has_model` in `solve`). The cold
                // basis was just installed via the per-element setters.
                // `if_values_pass = 0` requests a cold solve. The returned int is
                // the CLP solve status.
                unsafe { clp_ffi::cobre_clp_dual(self.handle, 0) }
            }
            ClpAlgorithm::Primal => {
                // SAFETY: as the dual arm — valid handle, model loaded, cold
                // basis installed. `if_values_pass = 0` requests a cold solve.
                unsafe { clp_ffi::cobre_clp_primal(self.handle, 0) }
            }
        }
    }

    /// Cold-solve escalation ladder run when the initial dual simplex falsely
    /// declares a feasible LP `PRIMAL_INFEASIBLE` (or stops short).
    ///
    /// CLP's bare dual simplex can spuriously report `PRIMAL_INFEASIBLE` on
    /// numerically delicate deep-stage LPs that are in fact feasible (proven:
    /// the failing stage moves with tolerances, disappears under the primal
    /// simplex, and the identical LP solves cleanly in `HiGHS`). This ladder
    /// re-solves the SAME already-loaded model (bounds plus any warm-start basis
    /// already installed) with progressively stronger settings, stopping at the
    /// first rung that returns OPTIMAL:
    ///
    /// 1. **Primal simplex** — the primal path alone clears the documented
    ///    repro.
    /// 2. **Perturbation on** (`cobre_clp_set_perturbation(50)`), then dual,
    ///    then primal — anti-cycling for degenerate vertices.
    /// 3. **Scaling on** (`cobre_clp_scaling(1)`), then dual, then primal —
    ///    rescales an ill-conditioned matrix.
    ///
    /// Every rung first resets to a clean all-slack cold basis
    /// ([`Self::escalate_run`]) so no rung inherits the failed basis from the
    /// initial dual solve. On OPTIMAL the solution is copied into the owned
    /// buffers (`copy_solution`) and an [`EscalationOutcome`] is returned by
    /// value so the caller can finish stats bookkeeping before borrowing those
    /// buffers. On exhaustion, `None` is returned and the caller surfaces the
    /// original error.
    ///
    /// The caller ([`SolverInterface::solve`]) is responsible for restoring the
    /// floor (deterministic) settings afterward by re-applying
    /// `current_profile`, regardless of outcome — perturbation and scaling are
    /// turned on here only for the duration of the ladder. The ladder is
    /// deterministic per-LP (fixed rung order, no randomness, no time-dependent
    /// branching), so results stay bit-for-bit identical across thread / rank
    /// counts within the CLP backend.
    pub(super) fn escalate_solve(&mut self) -> Option<EscalationOutcome> {
        // Rung order: (perturbation, scaling, algorithm). Perturbation `102`
        // disables CLP auto-perturbation (the deterministic floor); `50` turns
        // it on. Scaling `0` off, `1` on. The first OPTIMAL wins. This is a
        // fixed, allocation-free sequence (`LADDER_RUNGS` entries). Declared
        // first so no statements precede it (clippy::items_after_statements).
        const RUNGS: [(i32, i32, ClpAlgorithm); LADDER_RUNGS] = [
            // 1. Primal simplex (perturbation/scaling stay at the floor).
            (102, 0, ClpAlgorithm::Primal),
            // 2. Perturbation on, dual then primal.
            (50, 0, ClpAlgorithm::Dual),
            (50, 0, ClpAlgorithm::Primal),
            // 3. Scaling on (perturbation kept on from rung 2), dual then primal.
            (50, 1, ClpAlgorithm::Dual),
            (50, 1, ClpAlgorithm::Primal),
        ];

        let t0 = Instant::now();

        for (idx, &(perturbation, scaling, algorithm)) in RUNGS.iter().enumerate() {
            // SAFETY: `self.handle` is a valid, non-null CLP pointer with a model
            // loaded (asserted via `has_model` in `solve`). Both setters accept
            // any i32, retain no pointer, and cannot fail on a valid handle.
            unsafe {
                clp_ffi::cobre_clp_set_perturbation(self.handle, perturbation);
                clp_ffi::cobre_clp_scaling(self.handle, scaling);
            }

            let status = self.escalate_run(algorithm);

            if status == clp_ffi::CLP_STATUS_OPTIMAL {
                // SAFETY: `self.handle` is a valid, non-null CLP pointer that has
                // just been solved; iteration count is non-negative so the cast
                // is safe.
                #[allow(clippy::cast_sign_loss)]
                let iterations =
                    unsafe { clp_ffi::cobre_clp_number_iterations(self.handle) } as u64;
                // SAFETY: `self.handle` is a valid, non-null CLP pointer just
                // solved to optimality; objective is in minimize sense.
                let objective = unsafe { clp_ffi::cobre_clp_objective_value(self.handle) };

                // Copy the CLP-owned solution pointers into the owned buffers
                // immediately, before any further CLP call (pointers are valid
                // only until the next solve). Reuses the same buffers as the
                // happy path — no allocation.
                self.copy_solution();

                return Some(EscalationOutcome {
                    objective,
                    iterations,
                    solve_time: t0.elapsed().as_secs_f64(),
                    // `idx` is in `0..LADDER_RUNGS` (<= 5); `+1` cannot overflow
                    // and the widening to u64 is lossless.
                    attempts: idx as u64 + 1,
                });
            }
        }

        None
    }
}
