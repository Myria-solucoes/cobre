//! Integration tests for profile × retry-level tolerance composition.
//!
//! Verifies that primal and dual feasibility tolerances at retry levels 3, 7,
//! 10, and 11 satisfy `applied = max(level_default, profile_value)` (design
//! §5.5). Also verifies the iteration-cap composition at retry level 0 (AC-8).
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

#[cfg(feature = "test-support")]
mod tests {
    use cobre_solver::{HighsSolver, SolverInterface, StageTemplate};

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
        // Confirm via the trait setter that the profile value is 0.
        solver.set_simplex_iteration_limit_profile(0);

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
        solver.set_simplex_iteration_limit_profile(42_000);

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
        solver.set_ipm_iteration_limit_profile(500);

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

    /// Verify that `set_primal_feasibility_tolerance` writes the value to the
    /// underlying `HiGHS` option immediately (not only at retry time).
    #[test]
    fn primal_setter_propagates_to_ffi() {
        let mut solver = make_solver();
        solver.set_primal_feasibility_tolerance(3e-8);

        let tol = solver
            .get_double_option(c"primal_feasibility_tolerance")
            .expect("primal_feasibility_tolerance must be readable");

        assert!(
            (tol - 3e-8).abs() < 1e-20,
            "set_primal_feasibility_tolerance must write to FFI; expected 3e-8, got {tol}"
        );
    }

    // ── AC-9: loose profile (profile_value > level_default) ──────────────────
    //
    // Profile primal = dual = 1e-7.
    // Levels 3 and 7  have level_default = 1e-8  →  max(1e-8, 1e-7) = 1e-7.
    // Levels 10 and 11 have level_default = 1e-7  →  max(1e-7, 1e-7) = 1e-7.

    fn make_loose_profile_solver() -> HighsSolver {
        let mut solver = make_solver();
        solver.set_primal_feasibility_tolerance(1e-7);
        solver.set_dual_feasibility_tolerance(1e-7);
        solver
    }

    /// AC-9 — level 3, loose profile: applied tolerance must be 1e-7
    /// (= max(1e-8, 1e-7)) for both primal and dual.
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
            (primal - 1e-7).abs() < 1e-20,
            "level 3 loose: primal must be max(1e-8, 1e-7) = 1e-7; got {primal}"
        );
        assert!(
            (dual - 1e-7).abs() < 1e-20,
            "level 3 loose: dual must be max(1e-8, 1e-7) = 1e-7; got {dual}"
        );
    }

    /// AC-9 — level 7, loose profile: applied tolerance must be 1e-7
    /// (= max(1e-8, 1e-7)) for both primal and dual.
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
            (primal - 1e-7).abs() < 1e-20,
            "level 7 loose: primal must be max(1e-8, 1e-7) = 1e-7; got {primal}"
        );
        assert!(
            (dual - 1e-7).abs() < 1e-20,
            "level 7 loose: dual must be max(1e-8, 1e-7) = 1e-7; got {dual}"
        );
    }

    /// AC-9 — level 10, loose profile: applied tolerance must be 1e-7
    /// (= max(1e-7, 1e-7)) for both primal and dual.
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
            (primal - 1e-7).abs() < 1e-20,
            "level 10 loose: primal must be max(1e-7, 1e-7) = 1e-7; got {primal}"
        );
        assert!(
            (dual - 1e-7).abs() < 1e-20,
            "level 10 loose: dual must be max(1e-7, 1e-7) = 1e-7; got {dual}"
        );
    }

    /// AC-9 — level 11, loose profile: applied tolerance must be 1e-7
    /// (= max(1e-7, 1e-7)) for both primal and dual.
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
            (primal - 1e-7).abs() < 1e-20,
            "level 11 loose: primal must be max(1e-7, 1e-7) = 1e-7; got {primal}"
        );
        assert!(
            (dual - 1e-7).abs() < 1e-20,
            "level 11 loose: dual must be max(1e-7, 1e-7) = 1e-7; got {dual}"
        );
    }

    // ── AC-10: strict profile (profile_value < level_default) ────────────────
    //
    // Profile primal = dual = 1e-12.
    // Levels 3 and 7  have level_default = 1e-8  →  max(1e-8, 1e-12) = 1e-8.
    // Levels 10 and 11 have level_default = 1e-7  →  max(1e-7, 1e-12) = 1e-7.

    fn make_strict_profile_solver() -> HighsSolver {
        let mut solver = make_solver();
        solver.set_primal_feasibility_tolerance(1e-12);
        solver.set_dual_feasibility_tolerance(1e-12);
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
    // After `restore_default_settings()`, the tolerances reset to 1e-9 (the
    // hardcoded table values). `apply_profile_tolerances()` must then overwrite
    // them with the profile's values so HiGHS state and `current_profile` agree.

    /// Fix 1 — primal tolerance survives restore+profile sequence.
    ///
    /// Sets the profile primal tolerance to a non-default value (3e-8), then
    /// calls the combined `restore_defaults_then_apply_profile_for_test` helper
    /// (which mirrors the finalization path in `retry_escalation`). The FFI
    /// read-back must return 3e-8 (the profile value), not 1e-9 (the default
    /// table value), proving that `apply_profile_tolerances` wins.
    #[test]
    fn profile_primal_tolerance_restored_after_retry_finalization() {
        let mut solver = make_solver();
        solver.set_primal_feasibility_tolerance(3e-8);

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
    /// Same as the primal variant above, exercising the dual path.
    #[test]
    fn profile_dual_tolerance_restored_after_retry_finalization() {
        let mut solver = make_solver();
        solver.set_dual_feasibility_tolerance(5e-9);

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
        solver.set_ipm_iteration_limit_profile(0);

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
}
