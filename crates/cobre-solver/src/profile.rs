//! LP-solver configuration profiles.
//!
//! A [`SolveProfile`] is a set of LP-solver tuning values that callers swap in
//! at phase boundaries via `ProfiledSolver::set_profile`. All fields are
//! absolute — not relative to defaults. Two profiles can be compared with `==`
//! to decide whether any FFI calls are needed.

/// LP-solver configuration values applied at the start of each phase.
///
/// `SolveProfile` is a struct of tuning values that callers swap in via
/// `ProfiledSolver::set_profile` at phase boundaries. The profile defines
/// the configuration of the *default* solve attempt and the per-attempt
/// iteration cap; the retry ladder layers additional behavior on top.
///
/// All fields are absolute, not relative to defaults. Callers construct
/// profiles either via `Default::default()` (matching the historical
/// hardcoded behavior) or as `const` values for compile-time profile
/// libraries.
///
/// `SolveProfile` is `Copy` and `PartialEq` so the caller can compare the
/// desired profile against the currently-applied profile and skip FFI calls
/// when nothing has changed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveProfile {
    /// Primal feasibility tolerance used by the default solve attempt.
    ///
    /// Smaller values are stricter. The retry ladder applies
    /// `max(profile_value, level_default)` when it overrides this field, so a
    /// strict profile is never silently relaxed by an early retry level.
    pub primal_feasibility_tolerance: f64,

    /// Dual feasibility tolerance used by the default solve attempt.
    ///
    /// Same composition rules as `primal_feasibility_tolerance`.
    pub dual_feasibility_tolerance: f64,

    /// Per-attempt simplex iteration cap, applied to the default attempt and
    /// every retry level.
    ///
    /// A value of [`DEFAULT_PROFILE_HEURISTIC_SENTINEL`] (`0`) signals that
    /// the solver should fall back to its historical heuristic
    /// (`num_cols * 50 max 100_000` for `HighsSolver`). Any non-zero value is
    /// applied verbatim.
    ///
    /// Tighter values cause attempts to bail to the next retry strategy faster,
    /// bounding worst-case wall time per attempt at the cost of occasional
    /// escalation. Correctness is preserved as long as some retry level
    /// eventually solves the LP.
    pub simplex_iteration_limit: u32,

    /// Per-attempt IPM iteration cap.
    ///
    /// Applies to retry levels that invoke the interior-point solver. A value
    /// of `0` is treated as "unbounded" (no cap). Any positive value is applied
    /// verbatim.
    pub ipm_iteration_limit: u32,
}

impl Default for SolveProfile {
    /// Returns defaults that match the hardcoded behavior bit-for-bit,
    /// so that callers that never touch profiles see no behavioral change.
    ///
    /// | Field                          | Value                              |
    /// |--------------------------------|------------------------------------|
    /// | `primal_feasibility_tolerance` | `1e-6`                             |
    /// | `dual_feasibility_tolerance`   | `1e-6`                             |
    /// | `simplex_iteration_limit`      | `0` (use heuristic — see sentinel) |
    /// | `ipm_iteration_limit`          | `10_000`                           |
    ///
    /// The sentinel value `0` for `simplex_iteration_limit` causes `HighsSolver`
    /// to compute the historical per-call heuristic `num_cols * 50 max 100_000`
    /// rather than applying a flat cap.
    fn default() -> Self {
        Self {
            primal_feasibility_tolerance: 1e-6,
            dual_feasibility_tolerance: 1e-6,
            simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
            ipm_iteration_limit: 10_000,
        }
    }
}

/// Sentinel `u32` value indicating "use the historical heuristic
/// `num_cols * 50 max 100_000`" for the simplex iteration limit.
///
/// When [`SolveProfile::simplex_iteration_limit`] equals this value, solver
/// implementations MUST fall back to their per-call heuristic rather than
/// applying a flat iteration cap. Any non-zero value is applied verbatim as
/// the cap.
///
/// `0` is chosen as the sentinel because zero iterations is nonsensical for
/// any LP solver; legitimate iteration caps are always positive.
pub const DEFAULT_PROFILE_HEURISTIC_SENTINEL: u32 = 0;

/// Sentinel `u32` value indicating "unbounded" for the IPM iteration limit.
///
/// When [`SolveProfile::ipm_iteration_limit`] equals this value, solver
/// implementations MUST apply no cap (i.e., pass `i32::MAX` to the
/// underlying solver). Any positive value is applied verbatim as the cap.
///
/// `0` is chosen as the sentinel because zero IPM iterations is nonsensical;
/// legitimate iteration caps are always positive.
pub const DEFAULT_PROFILE_IPM_UNBOUNDED_SENTINEL: u32 = 0;

#[cfg(test)]
mod tests {
    use super::{DEFAULT_PROFILE_HEURISTIC_SENTINEL, SolveProfile};

    #[test]
    fn default_matches_historical_values() {
        let p = SolveProfile::default();
        assert_eq!(p.primal_feasibility_tolerance, 1e-6);
        assert_eq!(p.dual_feasibility_tolerance, 1e-6);
        assert_eq!(p.simplex_iteration_limit, 0);
        assert_eq!(p.ipm_iteration_limit, 10_000);
    }

    #[test]
    #[allow(clippy::clone_on_copy, clippy::no_effect_underscore_binding)]
    fn derived_traits_behave() {
        let p = SolveProfile::default();

        // Clone yields an equal value.
        let cloned = p.clone();
        assert_eq!(cloned, p);

        // Equality is reflexive.
        assert_eq!(p, p);

        // Copy semantics: binding p to q does not move p.
        let q = p;
        let _r = p; // This line would not compile if SolveProfile were not Copy.
        assert_eq!(q, p);

        // Debug output contains all four field names.
        let debug = format!("{p:?}");
        for field in [
            "primal_feasibility_tolerance",
            "dual_feasibility_tolerance",
            "simplex_iteration_limit",
            "ipm_iteration_limit",
        ] {
            assert!(
                debug.contains(field),
                "debug output missing '{field}': {debug}"
            );
        }
    }

    /// Verify that `SolveProfile` can be used in a `const` context.
    const _CONST_PROFILE: SolveProfile = SolveProfile {
        primal_feasibility_tolerance: 1e-6,
        dual_feasibility_tolerance: 1e-6,
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        ipm_iteration_limit: 10_000,
    };
}
