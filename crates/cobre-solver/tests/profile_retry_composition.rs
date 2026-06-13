//! Integration tests for profile × retry-level tolerance composition.
//!
//! Verifies that primal and dual feasibility tolerances at retry levels 3, 7,
//! 10, and 11 satisfy `applied = max(level_default, profile_value)`. Also
//! verifies the iteration-cap composition at retry level 0, and that the
//! profile is re-applied on every solve call (regression for the `HiGHS`
//! internal-reset bug).
//!
//! Requires the `test-support` feature:
//!   cargo nextest run -p cobre-solver --features test-support
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::float_cmp,
        clippy::too_many_lines,
        clippy::panic
    )
)]
#![cfg(feature = "highs")]

#[cfg(feature = "test-support")]
mod tests {
    use std::cell::Cell;

    use cobre_solver::types::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics};
    use cobre_solver::{HighsProfile, HighsSolver, ProfiledSolver, SolverInterface, StageTemplate};

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Minimal 2-column LP fixture.
    ///
    /// Two columns, no constraints.  Enough for `HiGHS` to load so that
    /// `num_cols = 2` is reflected in `set_iteration_limits` heuristics.
    ///
    ///   min  x0 + x1
    ///   x0 ∈ [0, 10],  x1 ∈ [0, 10]
    fn make_minimal_template() -> StageTemplate {
        StageTemplate {
            num_cols: 2,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32, 0, 0],
            row_indices: vec![],
            values: vec![],
            col_lower: vec![0.0, 0.0],
            col_upper: vec![10.0, 10.0],
            objective: vec![1.0, 1.0],
            row_lower: vec![],
            row_upper: vec![],
            n_state: 0,
            n_transfer: 0,
            n_dual_relevant: 0,
            n_hydro: 0,
            max_par_order: 0,
            col_scale: vec![],
            row_scale: vec![],
        }
    }

    /// Construct a fresh `HighsSolver` with the minimal 2-column LP loaded.
    fn make_solver() -> HighsSolver {
        let mut solver = HighsSolver::new().expect("HighsSolver::new() must succeed");
        solver.load_model(&make_minimal_template());
        solver
    }

    // ── AC-8: iteration-cap composition ──────────────────────────────────────

    /// AC-8 sentinel branch: when the profile simplex cap is the sentinel
    /// value (0), `apply_retry_level_options(0)` must apply the historical
    /// heuristic `max(100_000, num_cols * 50)`.
    ///
    /// With 2 columns the heuristic yields `max(100_000, 100) = 100_000`.
    #[test]
    fn sentinel_uses_heuristic() {
        let mut solver = make_solver();
        // Default profile has sentinel (0) for simplex_iteration_limit.
        // Confirm via apply_profile that the profile value is 0.
        solver.apply_profile(&HighsProfile {
            simplex_iteration_limit: 0,
            ..Default::default()
        });

        // Level 0 calls set_iteration_limits() internally.
        solver.apply_retry_level_options_for_test(0);

        let cap = solver
            .get_int_option(c"simplex_iteration_limit")
            .expect("simplex_iteration_limit must be readable");

        // num_cols = 2  →  heuristic = max(100_000, 2 * 50) = 100_000
        assert_eq!(
            cap, 100_000,
            "sentinel should produce heuristic cap 100_000; got {cap}"
        );
    }

    /// AC-8 literal-value branch: when the profile simplex cap is non-zero,
    /// `apply_retry_level_options(0)` must apply that exact value verbatim.
    #[test]
    fn nonzero_simplex_cap_used_verbatim() {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            simplex_iteration_limit: 42_000,
            ..Default::default()
        });

        // Level 0 calls set_iteration_limits() internally.
        solver.apply_retry_level_options_for_test(0);

        let cap = solver
            .get_int_option(c"simplex_iteration_limit")
            .expect("simplex_iteration_limit must be readable");

        assert_eq!(
            cap, 42_000,
            "non-zero profile cap must be applied verbatim; got {cap}"
        );
    }

    /// AC-8 IPM cap branch: `ipm_iteration_limit` from the profile is applied
    /// verbatim regardless of which retry level is active (`set_iteration_limits`
    /// always sets it).
    #[test]
    fn ipm_profile_value_applied() {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            ipm_iteration_limit: 500,
            ..Default::default()
        });

        // Level 0 calls set_iteration_limits() internally.
        solver.apply_retry_level_options_for_test(0);

        let cap = solver
            .get_int_option(c"ipm_iteration_limit")
            .expect("ipm_iteration_limit must be readable");

        assert_eq!(
            cap, 500,
            "profile IPM cap must be applied verbatim; got {cap}"
        );
    }

    // ── AC-8 bonus: default-attempt tolerance setter wires to FFI ────────────

    /// Verify that `apply_profile` writes the primal tolerance to the
    /// underlying `HiGHS` option immediately (not only at retry time).
    #[test]
    fn primal_setter_propagates_to_ffi() {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            primal_feasibility_tolerance: 3e-8,
            ..Default::default()
        });

        let tol = solver
            .get_double_option(c"primal_feasibility_tolerance")
            .expect("primal_feasibility_tolerance must be readable");

        assert!(
            (tol - 3e-8).abs() < 1e-20,
            "apply_profile must write primal tolerance to FFI; expected 3e-8, got {tol}"
        );
    }

    // ── AC-9: loose profile (profile_value > level_default) ──────────────────
    //
    // Profile primal = dual = 1e-5.
    // Levels 3 and 7  have level_default = 1e-8  →  max(1e-8, 1e-5) = 1e-5.
    // Levels 10 and 11 have level_default = 1e-7  →  max(1e-7, 1e-5) = 1e-5.

    fn make_loose_profile_solver() -> HighsSolver {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            primal_feasibility_tolerance: 1e-5,
            dual_feasibility_tolerance: 1e-5,
            ..Default::default()
        });
        solver
    }

    /// AC-9 — level 3, loose profile: applied tolerance must be 1e-5
    /// (= max(1e-8, 1e-5)) for both primal and dual.
    #[test]
    fn loose_profile_level3_applies_profile_value() {
        let mut solver = make_loose_profile_solver();
        solver.apply_retry_level_options_for_test(3);

        let primal = solver
            .get_double_option(c"primal_feasibility_tolerance")
            .expect("primal_feasibility_tolerance must be readable");
        let dual = solver
            .get_double_option(c"dual_feasibility_tolerance")
            .expect("dual_feasibility_tolerance must be readable");

        assert!(
            (primal - 1e-5).abs() < 1e-20,
            "level 3 loose: primal must be max(1e-8, 1e-5) = 1e-5; got {primal}"
        );
        assert!(
            (dual - 1e-5).abs() < 1e-20,
            "level 3 loose: dual must be max(1e-8, 1e-5) = 1e-5; got {dual}"
        );
    }

    /// AC-9 — level 7, loose profile: applied tolerance must be 1e-5
    /// (= max(1e-8, 1e-5)) for both primal and dual.
    #[test]
    fn loose_profile_level7_applies_profile_value() {
        let mut solver = make_loose_profile_solver();
        solver.apply_extended_retry_options_for_test(7);

        let primal = solver
            .get_double_option(c"primal_feasibility_tolerance")
            .expect("primal_feasibility_tolerance must be readable");
        let dual = solver
            .get_double_option(c"dual_feasibility_tolerance")
            .expect("dual_feasibility_tolerance must be readable");

        assert!(
            (primal - 1e-5).abs() < 1e-20,
            "level 7 loose: primal must be max(1e-8, 1e-5) = 1e-5; got {primal}"
        );
        assert!(
            (dual - 1e-5).abs() < 1e-20,
            "level 7 loose: dual must be max(1e-8, 1e-5) = 1e-5; got {dual}"
        );
    }

    /// AC-9 — level 10, loose profile: applied tolerance must be 1e-5
    /// (= max(1e-7, 1e-5)) for both primal and dual.
    #[test]
    fn loose_profile_level10_applies_profile_value() {
        let mut solver = make_loose_profile_solver();
        solver.apply_extended_retry_options_for_test(10);

        let primal = solver
            .get_double_option(c"primal_feasibility_tolerance")
            .expect("primal_feasibility_tolerance must be readable");
        let dual = solver
            .get_double_option(c"dual_feasibility_tolerance")
            .expect("dual_feasibility_tolerance must be readable");

        assert!(
            (primal - 1e-5).abs() < 1e-20,
            "level 10 loose: primal must be max(1e-7, 1e-5) = 1e-5; got {primal}"
        );
        assert!(
            (dual - 1e-5).abs() < 1e-20,
            "level 10 loose: dual must be max(1e-7, 1e-5) = 1e-5; got {dual}"
        );
    }

    /// AC-9 — level 11, loose profile: applied tolerance must be 1e-5
    /// (= max(1e-7, 1e-5)) for both primal and dual.
    #[test]
    fn loose_profile_level11_applies_profile_value() {
        let mut solver = make_loose_profile_solver();
        solver.apply_extended_retry_options_for_test(11);

        let primal = solver
            .get_double_option(c"primal_feasibility_tolerance")
            .expect("primal_feasibility_tolerance must be readable");
        let dual = solver
            .get_double_option(c"dual_feasibility_tolerance")
            .expect("dual_feasibility_tolerance must be readable");

        assert!(
            (primal - 1e-5).abs() < 1e-20,
            "level 11 loose: primal must be max(1e-7, 1e-5) = 1e-5; got {primal}"
        );
        assert!(
            (dual - 1e-5).abs() < 1e-20,
            "level 11 loose: dual must be max(1e-7, 1e-5) = 1e-5; got {dual}"
        );
    }

    // ── AC-10: strict profile (profile_value < level_default) ────────────────
    //
    // Profile primal = dual = 1e-12.
    // Levels 3 and 7  have level_default = 1e-8  →  max(1e-8, 1e-12) = 1e-8.
    // Levels 10 and 11 have level_default = 1e-7  →  max(1e-7, 1e-12) = 1e-7.

    fn make_strict_profile_solver() -> HighsSolver {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            primal_feasibility_tolerance: 1e-12,
            dual_feasibility_tolerance: 1e-12,
            ..Default::default()
        });
        solver
    }

    /// AC-10 — level 3, strict profile: applied tolerance must be 1e-8
    /// (= max(1e-8, 1e-12)) for both primal and dual.
    #[test]
    fn strict_profile_level3_applies_level_default() {
        let mut solver = make_strict_profile_solver();
        solver.apply_retry_level_options_for_test(3);

        let primal = solver
            .get_double_option(c"primal_feasibility_tolerance")
            .expect("primal_feasibility_tolerance must be readable");
        let dual = solver
            .get_double_option(c"dual_feasibility_tolerance")
            .expect("dual_feasibility_tolerance must be readable");

        assert!(
            (primal - 1e-8).abs() < 1e-20,
            "level 3 strict: primal must be max(1e-8, 1e-12) = 1e-8; got {primal}"
        );
        assert!(
            (dual - 1e-8).abs() < 1e-20,
            "level 3 strict: dual must be max(1e-8, 1e-12) = 1e-8; got {dual}"
        );
    }

    /// AC-10 — level 7, strict profile: applied tolerance must be 1e-8
    /// (= max(1e-8, 1e-12)) for both primal and dual.
    #[test]
    fn strict_profile_level7_applies_level_default() {
        let mut solver = make_strict_profile_solver();
        solver.apply_extended_retry_options_for_test(7);

        let primal = solver
            .get_double_option(c"primal_feasibility_tolerance")
            .expect("primal_feasibility_tolerance must be readable");
        let dual = solver
            .get_double_option(c"dual_feasibility_tolerance")
            .expect("dual_feasibility_tolerance must be readable");

        assert!(
            (primal - 1e-8).abs() < 1e-20,
            "level 7 strict: primal must be max(1e-8, 1e-12) = 1e-8; got {primal}"
        );
        assert!(
            (dual - 1e-8).abs() < 1e-20,
            "level 7 strict: dual must be max(1e-8, 1e-12) = 1e-8; got {dual}"
        );
    }

    /// AC-10 — level 10, strict profile: applied tolerance must be 1e-7
    /// (= max(1e-7, 1e-12)) for both primal and dual.
    #[test]
    fn strict_profile_level10_applies_level_default() {
        let mut solver = make_strict_profile_solver();
        solver.apply_extended_retry_options_for_test(10);

        let primal = solver
            .get_double_option(c"primal_feasibility_tolerance")
            .expect("primal_feasibility_tolerance must be readable");
        let dual = solver
            .get_double_option(c"dual_feasibility_tolerance")
            .expect("dual_feasibility_tolerance must be readable");

        assert!(
            (primal - 1e-7).abs() < 1e-20,
            "level 10 strict: primal must be max(1e-7, 1e-12) = 1e-7; got {primal}"
        );
        assert!(
            (dual - 1e-7).abs() < 1e-20,
            "level 10 strict: dual must be max(1e-7, 1e-12) = 1e-7; got {dual}"
        );
    }

    /// AC-10 — level 11, strict profile: applied tolerance must be 1e-7
    /// (= max(1e-7, 1e-12)) for both primal and dual.
    #[test]
    fn strict_profile_level11_applies_level_default() {
        let mut solver = make_strict_profile_solver();
        solver.apply_extended_retry_options_for_test(11);

        let primal = solver
            .get_double_option(c"primal_feasibility_tolerance")
            .expect("primal_feasibility_tolerance must be readable");
        let dual = solver
            .get_double_option(c"dual_feasibility_tolerance")
            .expect("dual_feasibility_tolerance must be readable");

        assert!(
            (primal - 1e-7).abs() < 1e-20,
            "level 11 strict: primal must be max(1e-7, 1e-12) = 1e-7; got {primal}"
        );
        assert!(
            (dual - 1e-7).abs() < 1e-20,
            "level 11 strict: dual must be max(1e-7, 1e-12) = 1e-7; got {dual}"
        );
    }

    // ── Fix 1: profile-aware restore after retry escalation ──────────────────
    //
    // After `restore_default_settings()`, the tolerances reset to 1e-6 (the
    // hardcoded table values). `apply_profile_tolerances()` must then overwrite
    // them with the profile's values so HiGHS state and `current_profile` agree.

    /// Fix 1 — primal tolerance survives restore+profile sequence.
    ///
    /// Sets the profile primal tolerance to a non-default value (3e-8), then
    /// calls the combined `restore_defaults_then_apply_profile_for_test` helper
    /// (which mirrors the finalization path in `retry_escalation`). The FFI
    /// read-back must return 3e-8 (the profile value), not 1e-6 (the default
    /// table value), proving that `apply_profile_tolerances` wins.
    #[test]
    fn profile_primal_tolerance_restored_after_retry_finalization() {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            primal_feasibility_tolerance: 3e-8,
            ..Default::default()
        });

        // Simulate the finalization path: restore defaults then re-apply profile.
        solver.restore_defaults_then_apply_profile_for_test();

        let tol = solver
            .get_double_option(c"primal_feasibility_tolerance")
            .expect("primal_feasibility_tolerance must be readable");

        assert!(
            (tol - 3e-8).abs() < 1e-20,
            "primal tolerance must be 3e-8 (profile value) after restore+profile, got {tol}"
        );
    }

    /// Fix 1 — dual tolerance survives restore+profile sequence.
    ///
    /// Sets the profile dual tolerance to 5e-9 (below the 1e-6 default),
    /// simulates the retry finalization path, and verifies the profile value wins.
    #[test]
    fn profile_dual_tolerance_restored_after_retry_finalization() {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            dual_feasibility_tolerance: 5e-9,
            ..Default::default()
        });

        // Simulate the finalization path: restore defaults then re-apply profile.
        solver.restore_defaults_then_apply_profile_for_test();

        let tol = solver
            .get_double_option(c"dual_feasibility_tolerance")
            .expect("dual_feasibility_tolerance must be readable");

        assert!(
            (tol - 5e-9).abs() < 1e-20,
            "dual tolerance must be 5e-9 (profile value) after restore+profile, got {tol}"
        );
    }

    // ── Fix 2: IPM iteration_limit sentinel for "unbounded" ──────────────────
    //
    // When `ipm_iteration_limit == 0` (DEFAULT_PROFILE_IPM_UNBOUNDED_SENTINEL),
    // `set_iteration_limits` must pass `i32::MAX` to HiGHS — not 0, which HiGHS
    // would interpret as "no iterations allowed".

    /// Fix 2 — `ipm_iteration_limit = 0` maps to `i32::MAX` at the FFI layer.
    ///
    /// Sets the profile IPM cap to 0 (the "unbounded" sentinel), triggers
    /// `set_iteration_limits` via `apply_retry_level_options(0)`, and reads
    /// back the `HiGHS` option. The value must be `i32::MAX`, not 0.
    #[test]
    fn ipm_sentinel_zero_maps_to_i32_max() {
        let mut solver = make_solver();
        // 0 is DEFAULT_PROFILE_IPM_UNBOUNDED_SENTINEL — "unbounded".
        solver.apply_profile(&HighsProfile {
            ipm_iteration_limit: 0,
            ..Default::default()
        });

        // Level 0 calls set_iteration_limits() internally.
        solver.apply_retry_level_options_for_test(0);

        let cap = solver
            .get_int_option(c"ipm_iteration_limit")
            .expect("ipm_iteration_limit must be readable");

        assert_eq!(
            cap,
            i32::MAX,
            "ipm_iteration_limit sentinel 0 must produce i32::MAX; got {cap}"
        );
    }

    // ── Delta-only profile dispatch: solve() does not re-apply ───────────────
    //
    // ProfiledSolver installs the active profile at phase boundaries via
    // set_profile (delta-gated by PartialEq) and re-applies it on the retry
    // path. solve() itself does NOT re-apply the profile: on the
    // forward/backward/simulation hot path the inner solver's options already
    // equal the active profile, so re-applying on every solve would issue
    // redundant FFI option setters. This test verifies that solves with no
    // intervening profile change dispatch zero additional apply_profile calls.

    /// Counts calls to `apply_profile`.
    struct SetterCountMock {
        apply_profile_call_count: Cell<usize>,
    }

    impl SetterCountMock {
        fn new() -> Self {
            Self {
                apply_profile_call_count: Cell::new(0),
            }
        }
    }

    // SAFETY: used only on a single thread within this test.
    unsafe impl Send for SetterCountMock {}

    impl SolverInterface for SetterCountMock {
        type Profile = HighsProfile;

        fn apply_profile(&mut self, _profile: &HighsProfile) {
            self.apply_profile_call_count
                .set(self.apply_profile_call_count.get() + 1);
        }

        fn load_model(&mut self, _t: &StageTemplate) {}
        fn add_rows(&mut self, _r: &RowBatch) {}
        fn set_row_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
        fn set_col_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
        fn solve(&mut self, _b: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
            Err(SolverError::InternalError {
                message: "mock".to_string(),
                error_code: None,
            })
        }
        fn get_basis(&mut self, _o: &mut Basis) {}
        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }
        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&SolverStatistics::default());
        }
        fn name(&self) -> &'static str {
            "SetterCountMock"
        }
        fn solver_name_version(&self) -> String {
            "SetterCountMock 0.0.0".to_string()
        }
    }

    /// Delta-only dispatch: once a profile is installed via `set_profile`,
    /// subsequent `solve()` calls with no intervening profile change dispatch
    /// no further `apply_profile` calls to the inner solver, and re-installing
    /// the identical profile is a no-op.
    #[test]
    fn solve_does_not_reapply_unchanged_profile() {
        let mock = SetterCountMock::new();
        let mut solver = ProfiledSolver::new(mock);

        // Apply a non-default profile once at phase entry (simulates phase boundary).
        let non_default = HighsProfile {
            primal_feasibility_tolerance: 1e-7,
            dual_feasibility_tolerance: 1e-7,
            simplex_iteration_limit: 50_000,
            ipm_iteration_limit: 5_000,
            simplex_dual_edge_weight_strategy: 0,
            simplex_scale_strategy: 0,
            simplex_price_strategy: 2,
        };
        solver.set_profile(&non_default);

        // set_profile dispatched one apply_profile call for the non-default profile.
        // Reset the counter so we only measure solve-triggered calls.
        solver.inner_mut().apply_profile_call_count.set(0);

        for _ in 0..3 {
            let _ = solver.solve(None);
        }

        // solve() must NOT re-apply an unchanged profile: the inner solver's
        // options already equal the active profile on the hot path.
        let solve_count = solver.inner().apply_profile_call_count.get();
        assert_eq!(
            solve_count, 0,
            "solve() must not re-apply an unchanged profile; expected 0, got {solve_count}"
        );

        // Re-installing the identical profile is a delta-gated no-op.
        solver.set_profile(&non_default);
        let after_reset = solver.inner().apply_profile_call_count.get();
        assert_eq!(
            after_reset, 0,
            "set_profile with an unchanged profile must be a no-op; expected 0, got {after_reset}"
        );
    }
}
