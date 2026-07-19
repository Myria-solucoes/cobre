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

    /// Minimal 2-column, no-constraint LP fixture. `num_cols = 2` feeds the
    /// `set_iteration_limits` heuristic the AC-8 tests check.
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
        solver.apply_profile(&HighsProfile {
            simplex_iteration_limit: 0,
            ..Default::default()
        });

        solver.apply_retry_level_options_for_test(0);

        let cap = solver
            .get_int_option(c"simplex_iteration_limit")
            .expect("simplex_iteration_limit must be readable");

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
    // hardcoded table values). `reapply_profile()` must then overwrite
    // them with the profile's values so HiGHS state and `current_profile` agree.

    /// Fix 1 — primal tolerance survives restore+profile sequence.
    ///
    /// Sets the profile primal tolerance to a non-default value (3e-8), then
    /// calls the combined `restore_defaults_then_reapply_profile_for_test` helper
    /// (which mirrors the finalization path in `retry_escalation`). The FFI
    /// read-back must return 3e-8 (the profile value), not 1e-6 (the default
    /// table value), proving that `reapply_profile` wins.
    #[test]
    fn profile_primal_tolerance_restored_after_retry_finalization() {
        let mut solver = make_solver();
        solver.apply_profile(&HighsProfile {
            primal_feasibility_tolerance: 3e-8,
            ..Default::default()
        });

        solver.restore_defaults_then_reapply_profile_for_test();

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

        solver.restore_defaults_then_reapply_profile_for_test();

        let tol = solver
            .get_double_option(c"dual_feasibility_tolerance")
            .expect("dual_feasibility_tolerance must be readable");

        assert!(
            (tol - 5e-9).abs() < 1e-20,
            "dual tolerance must be 5e-9 (profile value) after restore+profile, got {tol}"
        );
    }

    // ── Full-field profile-seam readback ─────────────────────────────────────
    //
    // Power statement: catches a `HighsProfile` field that `reapply_profile`
    // forgets to re-install after `restore_default_settings` — the seam the two
    // `Fix 1` tests above only sample via the tolerances. It does NOT catch a
    // HiGHS-internal option the profile never carries (only the fields
    // `reapply_profile` owns are read back below).

    /// Every field `reapply_profile` is responsible for — both feasibility
    /// tolerances and all three simplex strategy ints — survives
    /// `restore_default_settings` followed by `reapply_profile`, with every one
    /// of those fields installed at a non-default value simultaneously.
    #[test]
    fn full_profile_survives_retry_finalization_seam() {
        let mut solver = make_solver();
        let profile = HighsProfile {
            primal_feasibility_tolerance: 1e-7,
            dual_feasibility_tolerance: 1e-7,
            simplex_iteration_limit: 50_000,
            ipm_iteration_limit: 5_000,
            simplex_dual_edge_weight_strategy: 0, // Dantzig
            simplex_scale_strategy: 2,            // Curtis-Reid
            simplex_price_strategy: 2,            // RowHyperSparse
        };
        assert_ne!(
            profile,
            HighsProfile::default(),
            "fixture profile must differ from every HighsProfile::default() field"
        );
        solver.apply_profile(&profile);

        solver.restore_defaults_then_reapply_profile_for_test();

        let primal = solver
            .get_double_option(c"primal_feasibility_tolerance")
            .expect("primal_feasibility_tolerance must be readable");
        let dual = solver
            .get_double_option(c"dual_feasibility_tolerance")
            .expect("dual_feasibility_tolerance must be readable");
        let dual_edge_weight = solver
            .get_int_option(c"simplex_dual_edge_weight_strategy")
            .expect("simplex_dual_edge_weight_strategy must be readable");
        let scale = solver
            .get_int_option(c"simplex_scale_strategy")
            .expect("simplex_scale_strategy must be readable");
        let price = solver
            .get_int_option(c"simplex_price_strategy")
            .expect("simplex_price_strategy must be readable");

        assert!(
            (primal - profile.primal_feasibility_tolerance).abs() < 1e-20,
            "primal tolerance must survive the finalization seam; expected {}, got {primal}",
            profile.primal_feasibility_tolerance
        );
        assert!(
            (dual - profile.dual_feasibility_tolerance).abs() < 1e-20,
            "dual tolerance must survive the finalization seam; expected {}, got {dual}",
            profile.dual_feasibility_tolerance
        );
        assert_eq!(
            dual_edge_weight, profile.simplex_dual_edge_weight_strategy,
            "simplex_dual_edge_weight_strategy must survive the finalization seam"
        );
        assert_eq!(
            scale, profile.simplex_scale_strategy,
            "simplex_scale_strategy must survive the finalization seam"
        );
        assert_eq!(
            price, profile.simplex_price_strategy,
            "simplex_price_strategy must survive the finalization seam"
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
        solver.apply_profile(&HighsProfile {
            ipm_iteration_limit: 0,
            ..Default::default()
        });

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
    // ProfiledSolver installs the active profile at phase boundaries (delta-gated
    // by PartialEq) and on the retry path, but solve() does NOT re-apply it: on
    // the hot path the inner solver's options already equal the active profile, so
    // re-applying every solve would issue redundant FFI option setters.

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

        // Reset the counter `set_profile` just bumped, to measure only the
        // solve-triggered calls below.
        solver.inner_mut().apply_profile_call_count.set(0);

        for _ in 0..3 {
            let _ = solver.solve(None);
        }

        let solve_count = solver.inner().apply_profile_call_count.get();
        assert_eq!(
            solve_count, 0,
            "solve() must not re-apply an unchanged profile; expected 0, got {solve_count}"
        );

        solver.set_profile(&non_default);
        let after_reset = solver.inner().apply_profile_call_count.get();
        assert_eq!(
            after_reset, 0,
            "set_profile with an unchanged profile must be a no-op; expected 0, got {after_reset}"
        );
    }
}
