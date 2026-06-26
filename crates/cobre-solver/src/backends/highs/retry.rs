//! Warm-start retry-escalation ladder for [`HighsSolver`](super::HighsSolver).
//!
//! Governs the spurious-INFEASIBLE / spurious-UNBOUNDED recovery path and is
//! determinism-sensitive.

use std::time::Instant;

use super::solver::HighsSolver;
use crate::{ffi, types::SolverError};

/// Outcome of a successful retry escalation in [`HighsSolver::retry_escalation`].
pub(super) struct RetryOutcome {
    pub(super) attempts: u64,
    pub(super) solve_time: f64,
    pub(super) iterations: u64,
    /// The retry level (0..11) at which the solve succeeded.
    pub(super) level: u32,
}

impl HighsSolver {
    /// Run the 12-level retry escalation when the initial solve fails.
    ///
    /// Returns `Ok(RetryOutcome)` when a retry level finds optimal, or
    /// `Err((attempts, SolverError))` when all levels are exhausted or a
    /// terminal error is encountered. The caller is responsible for
    /// updating `self.stats` based on the outcome.
    ///
    /// Settings are always restored to defaults before returning (regardless
    /// of outcome).
    pub(super) fn retry_escalation(
        &mut self,
        is_unbounded: bool,
    ) -> Result<RetryOutcome, (u64, SolverError)> {
        // HiGHS `time_limit` is NOT used because HiGHS tracks elapsed time
        // cumulatively from instance creation — neither `clear_solver()` nor
        // option changes reset the internal timer. Iteration limits provide the
        // primary per-attempt safeguard; wall-clock budgets the secondary guard.
        let phase1_wall_budget = 15.0_f64;
        let phase2_wall_budget = 30.0_f64;
        let overall_budget = 120.0_f64;
        let num_retry_levels = 12_u32;

        let retry_start = Instant::now();
        let mut retry_attempts: u64 = 0;
        let mut terminal_err: Option<SolverError> = None;
        let mut found_optimal = false;
        let mut optimal_time = 0.0_f64;
        let mut optimal_iterations: u64 = 0;
        let mut optimal_level = 0_u32;

        for level in 0..num_retry_levels {
            if retry_start.elapsed().as_secs_f64() >= overall_budget {
                break;
            }

            self.apply_retry_level_options(level);

            retry_attempts += 1;

            let t_retry = Instant::now();
            let retry_status = self.run_once();
            let retry_time = t_retry.elapsed().as_secs_f64();

            if retry_status == ffi::HIGHS_MODEL_STATUS_OPTIMAL {
                // SAFETY: handle is valid non-null HiGHS pointer.
                #[allow(clippy::cast_sign_loss)]
                let iters =
                    unsafe { ffi::cobre_highs_get_simplex_iteration_count(self.handle) } as u64;
                found_optimal = true;
                optimal_time = retry_time;
                optimal_iterations = iters;
                optimal_level = level;
                break;
            }

            // UNBOUNDED / ITERATION_LIMIT / budget-exceeded continue (the
            // failure may be spurious and another strategy may converge);
            // other terminal statuses (INFEASIBLE) stop immediately.
            let level_budget = if level <= 4 {
                phase1_wall_budget
            } else {
                phase2_wall_budget
            };
            let budget_exceeded = retry_time > level_budget;
            let retryable = retry_status == ffi::HIGHS_MODEL_STATUS_UNBOUNDED
                || retry_status == ffi::HIGHS_MODEL_STATUS_ITERATION_LIMIT
                || budget_exceeded;
            if !retryable && let Some(e) = self.interpret_terminal_status(retry_status, retry_time)
            {
                terminal_err = Some(e);
                break;
            }
        }

        // Unconditional restore: `apply_profile_tolerances` re-applies the
        // caller's profile on top of `restore_default_settings` so HiGHS state
        // and `current_profile` stay in sync; retry-only options
        // (`user_objective_scale` / `user_bound_scale`) need explicit reset.
        self.restore_default_settings();
        self.apply_profile_tolerances();
        self.restore_iteration_limits();
        unsafe {
            ffi::cobre_highs_set_int_option(self.handle, c"user_objective_scale".as_ptr(), 0);
            ffi::cobre_highs_set_int_option(self.handle, c"user_bound_scale".as_ptr(), 0);
        }

        if found_optimal {
            return Ok(RetryOutcome {
                attempts: retry_attempts,
                solve_time: optimal_time,
                iterations: optimal_iterations,
                level: optimal_level,
            });
        }

        Err((
            retry_attempts,
            terminal_err.unwrap_or_else(|| {
                if is_unbounded {
                    SolverError::Unbounded
                } else {
                    SolverError::NumericalDifficulty {
                        message:
                            "HiGHS failed to reach optimality after all retry escalation levels"
                                .to_string(),
                    }
                }
            }),
        ))
    }

    /// Apply `HiGHS` options for a specific retry escalation level.
    ///
    /// Phase 1 (levels 0-4) is cumulative: each level adds options on top of
    /// the previous state. Both phases apply `time_limit` and iteration limits
    /// as safeguards against hanging on hard LPs.
    ///
    /// Phase 2 (levels 5-11) starts fresh each time with its own time limit.
    ///
    /// # Safety (internal)
    ///
    /// All FFI calls use `self.handle` which is a valid non-null `HiGHS` pointer.
    /// Option names and values are static C strings with no retained pointers.
    pub(super) fn apply_retry_level_options(&mut self, level: u32) {
        match level {
            // Re-enable dual-simplex cost perturbation: the default runs with it
            // off for warm-start performance, which can stall on degenerate
            // vertices; restoring the `HiGHS` default `1.0` is the cheapest
            // first-line intervention against cycling.
            0 => {
                unsafe {
                    ffi::cobre_highs_clear_solver(self.handle);
                    ffi::cobre_highs_set_double_option(
                        self.handle,
                        c"dual_simplex_cost_perturbation_multiplier".as_ptr(),
                        1.0,
                    );
                }
                self.set_iteration_limits();
            }
            1 => unsafe {
                ffi::cobre_highs_set_string_option(
                    self.handle,
                    c"presolve".as_ptr(),
                    c"on".as_ptr(),
                );
            },
            2 => unsafe {
                ffi::cobre_highs_set_int_option(self.handle, c"simplex_strategy".as_ptr(), 1);
            },
            // Applied value = max(1e-8, profile_value): a looser profile is
            // preserved while a tighter one falls back to the level's default.
            3 => {
                let primal = f64::max(1e-8, self.current_profile.primal_feasibility_tolerance);
                let dual = f64::max(1e-8, self.current_profile.dual_feasibility_tolerance);
                // SAFETY: handle is valid non-null HiGHS pointer; option names
                // are static C string literals; no retained pointers.
                unsafe {
                    ffi::cobre_highs_set_double_option(
                        self.handle,
                        c"primal_feasibility_tolerance".as_ptr(),
                        primal,
                    );
                    ffi::cobre_highs_set_double_option(
                        self.handle,
                        c"dual_feasibility_tolerance".as_ptr(),
                        dual,
                    );
                }
            }
            4 => unsafe {
                ffi::cobre_highs_set_string_option(
                    self.handle,
                    c"solver".as_ptr(),
                    c"ipm".as_ptr(),
                );
            },
            _ => self.apply_extended_retry_options(level),
        }
    }

    /// Apply Phase 2 extended retry strategy options for levels 5-11.
    ///
    /// Each level starts from restored defaults with presolve and iteration
    /// limits, then applies level-specific scaling, tolerance, and solver
    /// options. Wall-clock budgets are managed by the caller.
    pub(super) fn apply_extended_retry_options(&mut self, level: u32) {
        self.restore_default_settings();
        self.set_iteration_limits();
        // SAFETY: handle is valid non-null HiGHS pointer; option names/values
        // are static C strings; no retained pointers after call.
        unsafe {
            ffi::cobre_highs_set_string_option(self.handle, c"presolve".as_ptr(), c"on".as_ptr());
        }
        match level {
            // L5/L6: no scaler override — every level inherits the default
            // scaler (Off; cobre's offline prescaler conditions the matrix).
            5 => {}
            6 => unsafe {
                ffi::cobre_highs_set_int_option(self.handle, c"simplex_strategy".as_ptr(), 1);
            },
            7 => {
                let primal = f64::max(1e-8, self.current_profile.primal_feasibility_tolerance);
                let dual = f64::max(1e-8, self.current_profile.dual_feasibility_tolerance);
                // SAFETY: handle is valid non-null HiGHS pointer; option names
                // are static C string literals; no retained pointers.
                unsafe {
                    ffi::cobre_highs_set_double_option(
                        self.handle,
                        c"primal_feasibility_tolerance".as_ptr(),
                        primal,
                    );
                    ffi::cobre_highs_set_double_option(
                        self.handle,
                        c"dual_feasibility_tolerance".as_ptr(),
                        dual,
                    );
                }
            }
            8 => unsafe {
                ffi::cobre_highs_set_int_option(self.handle, c"user_objective_scale".as_ptr(), -10);
            },
            9 => unsafe {
                ffi::cobre_highs_set_int_option(self.handle, c"simplex_strategy".as_ptr(), 1);
                ffi::cobre_highs_set_int_option(self.handle, c"user_objective_scale".as_ptr(), -10);
                ffi::cobre_highs_set_int_option(self.handle, c"user_bound_scale".as_ptr(), -5);
            },
            10 => {
                let primal = f64::max(1e-7, self.current_profile.primal_feasibility_tolerance);
                let dual = f64::max(1e-7, self.current_profile.dual_feasibility_tolerance);
                // SAFETY: handle is valid non-null HiGHS pointer; option names
                // are static C string literals; no retained pointers.
                unsafe {
                    ffi::cobre_highs_set_int_option(
                        self.handle,
                        c"user_objective_scale".as_ptr(),
                        -13,
                    );
                    ffi::cobre_highs_set_int_option(self.handle, c"user_bound_scale".as_ptr(), -8);
                    ffi::cobre_highs_set_double_option(
                        self.handle,
                        c"primal_feasibility_tolerance".as_ptr(),
                        primal,
                    );
                    ffi::cobre_highs_set_double_option(
                        self.handle,
                        c"dual_feasibility_tolerance".as_ptr(),
                        dual,
                    );
                }
            }
            11 => {
                let primal = f64::max(1e-7, self.current_profile.primal_feasibility_tolerance);
                let dual = f64::max(1e-7, self.current_profile.dual_feasibility_tolerance);
                // SAFETY: handle is valid non-null HiGHS pointer; option names
                // are static C string literals; no retained pointers.
                unsafe {
                    ffi::cobre_highs_set_string_option(
                        self.handle,
                        c"solver".as_ptr(),
                        c"ipm".as_ptr(),
                    );
                    ffi::cobre_highs_set_int_option(
                        self.handle,
                        c"user_objective_scale".as_ptr(),
                        -10,
                    );
                    ffi::cobre_highs_set_int_option(self.handle, c"user_bound_scale".as_ptr(), -5);
                    ffi::cobre_highs_set_double_option(
                        self.handle,
                        c"primal_feasibility_tolerance".as_ptr(),
                        primal,
                    );
                    ffi::cobre_highs_set_double_option(
                        self.handle,
                        c"dual_feasibility_tolerance".as_ptr(),
                        dual,
                    );
                }
            }
            _ => unreachable!(),
        }
    }
}
