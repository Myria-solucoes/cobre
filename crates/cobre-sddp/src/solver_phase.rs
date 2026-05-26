//! SDDP algorithm phase enum and named solver-profile constants.
//!
//! Each [`Phase`] variant maps to a [`SolveProfile`] constant that configures
//! the LP solver for that phase. In v1 all three constants equal
//! [`SolveProfile::default()`] field-for-field, preserving bit-for-bit
//! behavioral parity with the historical hard-coded defaults. Future tuning
//! (particularly of [`BACKWARD_PROFILE`]) is out of scope for this release and
//! will happen in a follow-up.
//!
//! Compile-time assertions at the bottom of this module catch any future drift
//! between the named constants and the documented field values.

use cobre_solver::{DEFAULT_PROFILE_HEURISTIC_SENTINEL, SolveProfile};

/// The three algorithmic phases of the SDDP algorithm.
///
/// Each variant corresponds to a distinct phase of the training/simulation
/// loop. The solver can be configured differently per phase by calling
/// `ProfiledSolver::set_profile(phase.profile())` at phase entry.
///
/// `Phase` is `Copy + Eq` so it can be used as a `HashMap` key, in `match`
/// patterns, and stored cheaply by value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Forward pass: sampling trajectories by solving LPs from stage 1 to T.
    Forward,
    /// Backward pass: computing Benders cuts by solving LPs from stage T to 1.
    Backward,
    /// Policy simulation: evaluating the trained policy on out-of-sample
    /// scenarios.
    Simulation,
}

/// Solver profile applied during the SDDP forward pass.
///
/// In v1 this equals [`SolveProfile::default()`] field-for-field, locking in
/// behavioral parity with the historical hard-coded tolerances. Future tuning
/// of forward-pass solver settings will update this constant.
pub const FORWARD_PROFILE: SolveProfile = SolveProfile {
    primal_feasibility_tolerance: 1e-6,
    dual_feasibility_tolerance: 1e-6,
    simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    ipm_iteration_limit: 10_000,
};

/// Solver profile applied during the SDDP backward pass.
///
/// In v1 this equals [`SolveProfile::default()`] field-for-field, locking in
/// behavioral parity with the historical hard-coded tolerances. Future tuning
/// of backward-pass solver settings (the primary motivation for this
/// infrastructure) will update this constant.
pub const BACKWARD_PROFILE: SolveProfile = SolveProfile {
    primal_feasibility_tolerance: 1e-6,
    dual_feasibility_tolerance: 1e-6,
    simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    ipm_iteration_limit: 10_000,
};

/// Solver profile applied during policy simulation.
///
/// In v1 this equals [`SolveProfile::default()`] field-for-field, locking in
/// behavioral parity with the historical hard-coded tolerances. Future tuning
/// of simulation-phase solver settings will update this constant.
pub const SIMULATION_PROFILE: SolveProfile = SolveProfile {
    primal_feasibility_tolerance: 1e-6,
    dual_feasibility_tolerance: 1e-6,
    simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    ipm_iteration_limit: 10_000,
};

impl Phase {
    /// Returns the [`SolveProfile`] that should be applied when entering this
    /// phase.
    ///
    /// The returned profile is cheap to copy (`SolveProfile` is `Copy` with
    /// four scalar fields) and should be passed to
    /// `ProfiledSolver::set_profile` at phase entry.
    #[must_use]
    pub fn profile(self) -> SolveProfile {
        match self {
            Phase::Forward => FORWARD_PROFILE,
            Phase::Backward => BACKWARD_PROFILE,
            Phase::Simulation => SIMULATION_PROFILE,
        }
    }
}

// ── Compile-time drift guards ──────────────────────────────────────────────
//
// These assertions run at compile time and catch any future divergence between
// the named profile constants and their documented field values.  If a field
// is added to `SolveProfile`, the compiler will reject any const struct
// literal that does not include it, forcing an explicit update here.

const _: () = {
    assert!(FORWARD_PROFILE.primal_feasibility_tolerance == 1e-6);
    assert!(FORWARD_PROFILE.dual_feasibility_tolerance == 1e-6);
    assert!(FORWARD_PROFILE.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL);
    assert!(FORWARD_PROFILE.ipm_iteration_limit == 10_000);

    assert!(BACKWARD_PROFILE.primal_feasibility_tolerance == 1e-6);
    assert!(BACKWARD_PROFILE.dual_feasibility_tolerance == 1e-6);
    assert!(BACKWARD_PROFILE.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL);
    assert!(BACKWARD_PROFILE.ipm_iteration_limit == 10_000);

    assert!(SIMULATION_PROFILE.primal_feasibility_tolerance == 1e-6);
    assert!(SIMULATION_PROFILE.dual_feasibility_tolerance == 1e-6);
    assert!(SIMULATION_PROFILE.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL);
    assert!(SIMULATION_PROFILE.ipm_iteration_limit == 10_000);
};

#[cfg(test)]
mod tests {
    use cobre_solver::SolveProfile;

    use super::{BACKWARD_PROFILE, FORWARD_PROFILE, Phase, SIMULATION_PROFILE};

    /// Each named profile constant equals `SolveProfile::default()`
    /// field-by-field in v1.
    #[test]
    fn named_profiles_equal_solver_default_in_v1() {
        let default = SolveProfile::default();
        let check = |profile: &SolveProfile| {
            assert_eq!(
                profile.primal_feasibility_tolerance,
                default.primal_feasibility_tolerance
            );
            assert_eq!(
                profile.dual_feasibility_tolerance,
                default.dual_feasibility_tolerance
            );
            assert_eq!(
                profile.simplex_iteration_limit,
                default.simplex_iteration_limit
            );
            assert_eq!(profile.ipm_iteration_limit, default.ipm_iteration_limit);
        };
        check(&FORWARD_PROFILE);
        check(&BACKWARD_PROFILE);
        check(&SIMULATION_PROFILE);
    }

    /// `Phase::profile()` returns the matching named constant for each
    /// variant.
    #[test]
    fn phase_profile_returns_matching_constant() {
        assert_eq!(Phase::Forward.profile(), FORWARD_PROFILE);
        assert_eq!(Phase::Backward.profile(), BACKWARD_PROFILE);
        assert_eq!(Phase::Simulation.profile(), SIMULATION_PROFILE);
    }
}
