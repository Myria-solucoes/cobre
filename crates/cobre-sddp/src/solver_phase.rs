//! SDDP algorithm phase enum and named solver-profile constants.
//!
//! Each [`Phase`] variant maps to a [`HighsProfile`] constant that configures
//! the LP solver for that phase. [`FORWARD_PROFILE`] and
//! [`SIMULATION_PROFILE`] match [`HighsProfile::default()`] field-for-field.
//! [`BACKWARD_PROFILE`] overrides one option — `simplex_price_strategy`
//! (`RowHyperSparse`, value `2`) — to exploit sparse cut subgradient rows
//! that dominate the backward LP row count. Edge-weight strategy stays at
//! Devex (the [`HighsProfile::default`] value) after empirical sweeps
//! showed Dantzig and `SteepestEdge` alternatives net worse on wall time
//! and tail latency respectively.
//!
//! Compile-time assertions at the bottom of this module catch any future drift
//! between the named constants and the documented field values.

use cobre_solver::{DEFAULT_PROFILE_HEURISTIC_SENTINEL, HighsProfile};

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
/// All fields equal [`HighsProfile::default()`], which uses the tightened
/// `1e-9` primal/dual feasibility tolerances.
pub const FORWARD_PROFILE: HighsProfile = HighsProfile {
    primal_feasibility_tolerance: 1e-9,
    dual_feasibility_tolerance: 1e-9,
    simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    ipm_iteration_limit: 10_000,
    simplex_dual_edge_weight_strategy: 1, // Devex (default)
    simplex_scale_strategy: 4,            // Equilibration (default)
    simplex_price_strategy: 1,            // Row (default)
};

/// Solver profile applied during the SDDP backward pass.
///
/// Overrides one field relative to [`HighsProfile::default()`]:
///
/// | Field                      | Default | Backward | Rationale                                          |
/// |----------------------------|---------|----------|----------------------------------------------------|
/// | `simplex_price_strategy`   | `1`     | `2`      | `RowHyperSparse`: exploits sparsity on backward LPs |
///
/// See `docs/design/highs-backward-pass-tuning.md` for the empirical
/// sweep that selected this minimal override.
pub const BACKWARD_PROFILE: HighsProfile = HighsProfile {
    primal_feasibility_tolerance: 1e-9,
    dual_feasibility_tolerance: 1e-9,
    simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    ipm_iteration_limit: 10_000,
    simplex_dual_edge_weight_strategy: 1, // Devex (matches default)
    simplex_scale_strategy: 4,            // Equilibration (matches default)
    simplex_price_strategy: 2,            // RowHyperSparse override
};

/// Solver profile applied during policy simulation.
///
/// All fields equal [`HighsProfile::default()`], which uses the tightened
/// `1e-9` primal/dual feasibility tolerances.
pub const SIMULATION_PROFILE: HighsProfile = HighsProfile {
    primal_feasibility_tolerance: 1e-9,
    dual_feasibility_tolerance: 1e-9,
    simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    ipm_iteration_limit: 10_000,
    simplex_dual_edge_weight_strategy: 1, // Devex (default)
    simplex_scale_strategy: 4,            // Equilibration (default)
    simplex_price_strategy: 1,            // Row (default)
};

impl Phase {
    /// Returns the [`HighsProfile`] that should be applied when entering this
    /// phase.
    ///
    /// The returned profile is cheap to copy (`HighsProfile` is `Copy`) and
    /// should be passed to `ProfiledSolver::set_profile` at phase entry.
    #[must_use]
    pub fn profile(self) -> HighsProfile {
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
// the named profile constants and their documented field values. If a field is
// added to `HighsProfile`, the compiler will reject any const struct literal
// that does not include it, forcing an explicit update here.

const _: () = {
    // FORWARD_PROFILE — all fields equal HighsProfile::default()
    assert!(FORWARD_PROFILE.primal_feasibility_tolerance == 1e-9);
    assert!(FORWARD_PROFILE.dual_feasibility_tolerance == 1e-9);
    assert!(FORWARD_PROFILE.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL);
    assert!(FORWARD_PROFILE.ipm_iteration_limit == 10_000);
    assert!(FORWARD_PROFILE.simplex_dual_edge_weight_strategy == 1);
    assert!(FORWARD_PROFILE.simplex_scale_strategy == 4);
    assert!(FORWARD_PROFILE.simplex_price_strategy == 1);

    // BACKWARD_PROFILE — price=2 (RowHyperSparse), other fields equal default
    assert!(BACKWARD_PROFILE.primal_feasibility_tolerance == 1e-9);
    assert!(BACKWARD_PROFILE.dual_feasibility_tolerance == 1e-9);
    assert!(BACKWARD_PROFILE.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL);
    assert!(BACKWARD_PROFILE.ipm_iteration_limit == 10_000);
    assert!(BACKWARD_PROFILE.simplex_dual_edge_weight_strategy == 1);
    assert!(BACKWARD_PROFILE.simplex_scale_strategy == 4);
    assert!(BACKWARD_PROFILE.simplex_price_strategy == 2);

    // SIMULATION_PROFILE — all fields equal HighsProfile::default()
    assert!(SIMULATION_PROFILE.primal_feasibility_tolerance == 1e-9);
    assert!(SIMULATION_PROFILE.dual_feasibility_tolerance == 1e-9);
    assert!(SIMULATION_PROFILE.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL);
    assert!(SIMULATION_PROFILE.ipm_iteration_limit == 10_000);
    assert!(SIMULATION_PROFILE.simplex_dual_edge_weight_strategy == 1);
    assert!(SIMULATION_PROFILE.simplex_scale_strategy == 4);
    assert!(SIMULATION_PROFILE.simplex_price_strategy == 1);
};

#[cfg(test)]
mod tests {
    use cobre_solver::HighsProfile;

    use super::{BACKWARD_PROFILE, FORWARD_PROFILE, Phase, SIMULATION_PROFILE};

    /// `Phase::profile()` returns the matching named constant for each variant.
    #[test]
    fn phase_profile_returns_matching_constant() {
        assert_eq!(Phase::Forward.profile(), FORWARD_PROFILE);
        assert_eq!(Phase::Backward.profile(), BACKWARD_PROFILE);
        assert_eq!(Phase::Simulation.profile(), SIMULATION_PROFILE);
    }

    /// Forward and simulation profiles equal [`HighsProfile::default()`];
    /// backward profile differs in two fields.
    #[test]
    fn forward_and_simulation_equal_default() {
        let default = HighsProfile::default();
        assert_eq!(FORWARD_PROFILE, default);
        assert_eq!(SIMULATION_PROFILE, default);
    }

    #[test]
    fn backward_profile_overrides_only_price_strategy() {
        let default = HighsProfile::default();
        assert_ne!(BACKWARD_PROFILE, default);
        assert_eq!(BACKWARD_PROFILE.simplex_price_strategy, 2);
        // Fields that are NOT overridden must match the default.
        assert_eq!(
            BACKWARD_PROFILE.simplex_dual_edge_weight_strategy,
            default.simplex_dual_edge_weight_strategy
        );
        assert_eq!(
            BACKWARD_PROFILE.primal_feasibility_tolerance,
            default.primal_feasibility_tolerance
        );
        assert_eq!(
            BACKWARD_PROFILE.dual_feasibility_tolerance,
            default.dual_feasibility_tolerance
        );
        assert_eq!(
            BACKWARD_PROFILE.simplex_iteration_limit,
            default.simplex_iteration_limit
        );
        assert_eq!(
            BACKWARD_PROFILE.ipm_iteration_limit,
            default.ipm_iteration_limit
        );
        assert_eq!(
            BACKWARD_PROFILE.simplex_scale_strategy,
            default.simplex_scale_strategy
        );
    }
}
