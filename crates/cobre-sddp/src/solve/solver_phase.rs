//! SDDP algorithm phase enum, the backend-agnostic [`PhaseProfiles`] trait, and
//! the per-phase named solver-profile constants.
//!
//! All three phases solve the same cut-laden LPs, so they share one tuned
//! deep-cut-pool profile. For `HiGHS`, [`FORWARD_PROFILE`], [`BACKWARD_PROFILE`],
//! and [`SIMULATION_PROFILE`] are identical: each overrides only
//! `simplex_price_strategy` (`RowHyperSparse`, value `2`) relative to
//! [`HighsProfile::default()`] to exploit the sparse cut-subgradient rows. The
//! CLP backend pins full dual steepest-edge pricing (`dual_pricing_mode = 1`)
//! and `factorization_frequency = 200`. All values come from an empirical solver
//! sweep on production-scale cases.
//!
//! ## Why the per-phase profiles live here, not in `cobre-solver`
//!
//! The mapping "which phase wants which solver behaviour" is algorithm knowledge
//! that must live in the algorithm crate; `cobre-solver` stays strictly
//! backend-agnostic and cannot know about SDDP phases. The [`PhaseProfiles`]
//! trait abstracts only *which phase is running*, never the tuning content. The
//! trait is local and the implemented profile types are foreign, which the
//! orphan rule permits.

#[cfg(feature = "highs")]
use cobre_solver::HighsProfile;
use cobre_solver::{ActiveProfile, DEFAULT_PROFILE_HEURISTIC_SENTINEL};
#[cfg(feature = "clp")]
use cobre_solver::{ClpAlgorithm, ClpProfile};

/// The three algorithmic phases of the SDDP algorithm.
///
/// The solver is configured per phase via
/// `ProfiledSolver::set_profile(phase.profile())` at phase entry.
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

/// Per-phase identity selection of a backend solver profile (see the module
/// docs for why the tuned values live in `cobre-sddp`, not `cobre-solver`).
///
/// The members are **associated constants** because every backend profile field
/// is const-constructible.
pub trait PhaseProfiles: Sized {
    /// Profile applied when entering the forward pass.
    const FORWARD: Self;
    /// Profile applied when entering the backward pass.
    const BACKWARD: Self;
    /// Profile applied when entering policy simulation.
    const SIMULATION: Self;
}

/// Per-phase `HighsProfile` selection, bit-for-bit equal to the
/// [`FORWARD_PROFILE`]/[`BACKWARD_PROFILE`]/[`SIMULATION_PROFILE`] constants.
///
/// The full field literals are written out because associated constants cannot
/// use the `..HighsProfile::default()` struct-update spread in const context.
#[cfg(feature = "highs")]
impl PhaseProfiles for HighsProfile {
    const FORWARD: Self = HighsProfile {
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        ipm_iteration_limit: 10_000,
        simplex_dual_edge_weight_strategy: 1, // Devex
        simplex_scale_strategy: 0,            // Off — cobre prescaler conditions the matrix
        simplex_price_strategy: 2,            // RowHyperSparse
    };
    const BACKWARD: Self = HighsProfile {
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        ipm_iteration_limit: 10_000,
        simplex_dual_edge_weight_strategy: 1, // Devex
        simplex_scale_strategy: 0,            // Off — cobre prescaler conditions the matrix
        simplex_price_strategy: 2,            // RowHyperSparse
    };
    const SIMULATION: Self = HighsProfile {
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        ipm_iteration_limit: 10_000,
        simplex_dual_edge_weight_strategy: 1, // Devex
        simplex_scale_strategy: 0,            // Off — cobre prescaler conditions the matrix
        simplex_price_strategy: 2,            // RowHyperSparse
    };
}

/// Per-phase `ClpProfile` selection.
///
/// `FORWARD` and `BACKWARD` share the tuned deep-cut-pool profile
/// (`dual_pricing_mode = 1`, `factorization_frequency = 200`). These are
/// CLP-native values for CLP's own option surface — **not** a translation of the
/// `HiGHS` per-phase profiles. `SIMULATION` keeps those values but selects the
/// **primal** simplex (see the const's comment for why dual fails there).
///
/// The full field literals are written out because associated constants cannot
/// use the `..ClpProfile::default()` struct-update spread in const context.
#[cfg(feature = "clp")]
impl PhaseProfiles for ClpProfile {
    const FORWARD: Self = ClpProfile {
        perturbation: 102,
        scaling: 0,
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        algorithm: ClpAlgorithm::Dual,
        dual_pricing_mode: 1,         // full dual steepest-edge
        factorization_frequency: 200, // refactor cadence
    };
    const BACKWARD: Self = ClpProfile {
        perturbation: 102,
        scaling: 0,
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        algorithm: ClpAlgorithm::Dual,
        dual_pricing_mode: 1,         // full dual steepest-edge
        factorization_frequency: 200, // refactor cadence
    };
    // Simulation runs the PRIMAL simplex, not dual: CLP's dual falsely declares
    // these warm-started, fully-frozen cut-laden simulation LPs infeasible, while
    // the primal simplex solves them directly and deterministically (so
    // bit-for-bit reproducibility holds). The dual-row pivot rule
    // (`dual_pricing_mode`) is unused by primal.
    const SIMULATION: Self = ClpProfile {
        perturbation: 102,
        scaling: 0,
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        algorithm: ClpAlgorithm::Primal,
        dual_pricing_mode: 1,         // full dual steepest-edge
        factorization_frequency: 200, // refactor cadence
    };
}

/// Solver profile applied during the SDDP forward pass — the tuned
/// deep-cut-pool profile (see the module docs); equal to [`BACKWARD_PROFILE`]
/// and [`SIMULATION_PROFILE`].
#[cfg(feature = "highs")]
pub const FORWARD_PROFILE: HighsProfile = HighsProfile {
    primal_feasibility_tolerance: 1e-9,
    dual_feasibility_tolerance: 1e-9,
    simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    ipm_iteration_limit: 10_000,
    simplex_dual_edge_weight_strategy: 1, // Devex
    simplex_scale_strategy: 0,            // Off — cobre prescaler conditions the matrix
    simplex_price_strategy: 2,            // RowHyperSparse
};

/// Solver profile applied during the SDDP backward pass — the tuned
/// deep-cut-pool profile (see the module docs); equal to [`FORWARD_PROFILE`]
/// and [`SIMULATION_PROFILE`].
#[cfg(feature = "highs")]
pub const BACKWARD_PROFILE: HighsProfile = HighsProfile {
    primal_feasibility_tolerance: 1e-9,
    dual_feasibility_tolerance: 1e-9,
    simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    ipm_iteration_limit: 10_000,
    simplex_dual_edge_weight_strategy: 1, // Devex
    simplex_scale_strategy: 0,            // Off — cobre prescaler conditions the matrix
    simplex_price_strategy: 2,            // RowHyperSparse
};

/// Solver profile applied during policy simulation — the tuned deep-cut-pool
/// profile (see the module docs); equal to [`FORWARD_PROFILE`] and
/// [`BACKWARD_PROFILE`].
#[cfg(feature = "highs")]
pub const SIMULATION_PROFILE: HighsProfile = HighsProfile {
    primal_feasibility_tolerance: 1e-9,
    dual_feasibility_tolerance: 1e-9,
    simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    ipm_iteration_limit: 10_000,
    simplex_dual_edge_weight_strategy: 1, // Devex
    simplex_scale_strategy: 0,            // Off — cobre prescaler conditions the matrix
    simplex_price_strategy: 2,            // RowHyperSparse
};

impl Phase {
    /// Returns the [`ActiveProfile`] to apply when entering this
    /// phase, delegating to the active backend's [`PhaseProfiles`] impl. Pass it
    /// to `ProfiledSolver::set_profile` at phase entry.
    #[must_use]
    pub fn profile(self) -> ActiveProfile {
        match self {
            Phase::Forward => <ActiveProfile as PhaseProfiles>::FORWARD,
            Phase::Backward => <ActiveProfile as PhaseProfiles>::BACKWARD,
            Phase::Simulation => <ActiveProfile as PhaseProfiles>::SIMULATION,
        }
    }
}

// Compile-time drift guard: a field added to `HighsProfile` makes the compiler
// reject these const literals until they are updated, keeping the named
// constants and their documented field values in sync.
#[cfg(feature = "highs")]
const _: () = {
    assert!(FORWARD_PROFILE.primal_feasibility_tolerance == 1e-9);
    assert!(FORWARD_PROFILE.dual_feasibility_tolerance == 1e-9);
    assert!(FORWARD_PROFILE.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL);
    assert!(FORWARD_PROFILE.ipm_iteration_limit == 10_000);
    assert!(FORWARD_PROFILE.simplex_dual_edge_weight_strategy == 1);
    assert!(FORWARD_PROFILE.simplex_scale_strategy == 0);
    assert!(FORWARD_PROFILE.simplex_price_strategy == 2);

    assert!(BACKWARD_PROFILE.primal_feasibility_tolerance == 1e-9);
    assert!(BACKWARD_PROFILE.dual_feasibility_tolerance == 1e-9);
    assert!(BACKWARD_PROFILE.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL);
    assert!(BACKWARD_PROFILE.ipm_iteration_limit == 10_000);
    assert!(BACKWARD_PROFILE.simplex_dual_edge_weight_strategy == 1);
    assert!(BACKWARD_PROFILE.simplex_scale_strategy == 0);
    assert!(BACKWARD_PROFILE.simplex_price_strategy == 2);

    assert!(SIMULATION_PROFILE.primal_feasibility_tolerance == 1e-9);
    assert!(SIMULATION_PROFILE.dual_feasibility_tolerance == 1e-9);
    assert!(SIMULATION_PROFILE.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL);
    assert!(SIMULATION_PROFILE.ipm_iteration_limit == 10_000);
    assert!(SIMULATION_PROFILE.simplex_dual_edge_weight_strategy == 1);
    assert!(SIMULATION_PROFILE.simplex_scale_strategy == 0);
    assert!(SIMULATION_PROFILE.simplex_price_strategy == 2);

    assert!(matches!(
        <HighsProfile as PhaseProfiles>::FORWARD.simplex_price_strategy,
        2
    ));
    assert!(matches!(
        <HighsProfile as PhaseProfiles>::BACKWARD.simplex_price_strategy,
        2
    ));
    assert!(matches!(
        <HighsProfile as PhaseProfiles>::SIMULATION.simplex_price_strategy,
        2
    ));
};

#[cfg(all(test, feature = "highs"))]
mod highs_tests {
    use cobre_solver::HighsProfile;

    use super::{BACKWARD_PROFILE, FORWARD_PROFILE, Phase, PhaseProfiles, SIMULATION_PROFILE};

    /// `Phase::profile()` returns the matching named constant for each variant.
    #[test]
    fn phase_profile_returns_matching_constant() {
        assert_eq!(Phase::Forward.profile(), FORWARD_PROFILE);
        assert_eq!(Phase::Backward.profile(), BACKWARD_PROFILE);
        assert_eq!(Phase::Simulation.profile(), SIMULATION_PROFILE);
    }

    /// Forward and simulation named constants equal the tuned deep-cut-pool
    /// profile (`BACKWARD_PROFILE`) and differ from [`HighsProfile::default()`]
    /// only in `simplex_price_strategy` (`2`, `RowHyperSparse`).
    #[test]
    fn forward_simulation_equal_tuned_profile() {
        let default = HighsProfile::default();
        assert_eq!(FORWARD_PROFILE, BACKWARD_PROFILE);
        assert_eq!(SIMULATION_PROFILE, BACKWARD_PROFILE);
        assert_ne!(FORWARD_PROFILE, default);
        assert_ne!(SIMULATION_PROFILE, default);
        assert_eq!(FORWARD_PROFILE.simplex_price_strategy, 2);
        assert_eq!(SIMULATION_PROFILE.simplex_price_strategy, 2);
        assert_eq!(default.simplex_price_strategy, 1);
        // Pinning the price strategy back to the default recovers the default.
        let mut forward_relaxed = FORWARD_PROFILE;
        forward_relaxed.simplex_price_strategy = default.simplex_price_strategy;
        assert_eq!(forward_relaxed, default);
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

    /// The `PhaseProfiles` impl's `FORWARD`/`SIMULATION` equal the tuned
    /// deep-cut-pool profile (`BACKWARD`) and differ from the default only in
    /// `simplex_price_strategy` (`2`, `RowHyperSparse`).
    #[test]
    fn phase_profiles_forward_simulation_equal_tuned_profile() {
        let default = HighsProfile::default();
        let forward = <HighsProfile as PhaseProfiles>::FORWARD;
        let simulation = <HighsProfile as PhaseProfiles>::SIMULATION;
        let backward = <HighsProfile as PhaseProfiles>::BACKWARD;
        // All three per-phase profiles are now identical (the tuned profile).
        assert_eq!(forward, backward);
        assert_eq!(simulation, backward);
        // They differ from the default only in the tuned price strategy.
        assert_ne!(forward, default);
        assert_ne!(simulation, default);
        assert_eq!(forward.simplex_price_strategy, 2);
        assert_eq!(simulation.simplex_price_strategy, 2);
        assert_eq!(default.simplex_price_strategy, 1);
    }

    /// The `PhaseProfiles` impl's `BACKWARD` overrides only
    /// `simplex_price_strategy` to `2`, matching the default elsewhere.
    #[test]
    fn phase_profiles_backward_overrides_only_price_strategy() {
        let default = HighsProfile::default();
        let backward = <HighsProfile as PhaseProfiles>::BACKWARD;
        assert_ne!(backward, default);
        assert_eq!(backward.simplex_price_strategy, 2);
        assert_eq!(
            backward.simplex_dual_edge_weight_strategy,
            default.simplex_dual_edge_weight_strategy
        );
        assert_eq!(
            backward.primal_feasibility_tolerance,
            default.primal_feasibility_tolerance
        );
        assert_eq!(
            backward.dual_feasibility_tolerance,
            default.dual_feasibility_tolerance
        );
        assert_eq!(
            backward.simplex_iteration_limit,
            default.simplex_iteration_limit
        );
        assert_eq!(backward.ipm_iteration_limit, default.ipm_iteration_limit);
        assert_eq!(
            backward.simplex_scale_strategy,
            default.simplex_scale_strategy
        );
    }

    /// Numeric inertness: every `PhaseProfiles` value is bit-for-bit equal to
    /// the corresponding current named constant.
    #[test]
    fn phase_profiles_bit_for_bit_match_named_constants() {
        assert_eq!(<HighsProfile as PhaseProfiles>::FORWARD, FORWARD_PROFILE);
        assert_eq!(<HighsProfile as PhaseProfiles>::BACKWARD, BACKWARD_PROFILE);
        assert_eq!(
            <HighsProfile as PhaseProfiles>::SIMULATION,
            SIMULATION_PROFILE
        );
    }
}

#[cfg(all(test, feature = "clp"))]
mod clp_tests {
    use cobre_solver::{ClpAlgorithm, ClpProfile};

    use super::{Phase, PhaseProfiles};

    /// The CLP `FORWARD` and `BACKWARD` profiles are the identical tuned
    /// deep-cut-pool profile (dual simplex, `dual_pricing_mode = 1`,
    /// `factorization_frequency = 200`). `SIMULATION` keeps those tuned values
    /// but selects the **primal** simplex, so it differs from `BACKWARD` in
    /// `algorithm` alone. All three differ from [`ClpProfile::default()`].
    #[test]
    fn clp_phase_profiles_tuned_with_primal_simulation() {
        let default = ClpProfile::default();
        let forward = <ClpProfile as PhaseProfiles>::FORWARD;
        let simulation = <ClpProfile as PhaseProfiles>::SIMULATION;
        let backward = <ClpProfile as PhaseProfiles>::BACKWARD;
        // FORWARD and BACKWARD are the identical tuned (dual) profile.
        assert_eq!(forward, backward);
        // SIMULATION runs the primal simplex to dodge CLP's dual
        // false-infeasibilities on warm-started, cut-laden simulation LPs, so it
        // differs from the tuned profile in `algorithm` ONLY.
        assert_ne!(simulation, backward);
        assert_eq!(forward.algorithm, ClpAlgorithm::Dual);
        assert_eq!(backward.algorithm, ClpAlgorithm::Dual);
        assert_eq!(simulation.algorithm, ClpAlgorithm::Primal);
        assert_eq!(
            ClpProfile {
                algorithm: ClpAlgorithm::Dual,
                ..simulation
            },
            backward,
            "SIMULATION must equal the tuned profile except for the primal algorithm"
        );
        // They differ from the default only in the tuned fields.
        assert_ne!(forward, default);
        assert_ne!(simulation, default);
        assert_eq!(forward.dual_pricing_mode, 1);
        assert_eq!(forward.factorization_frequency, 200);
        assert_eq!(simulation.dual_pricing_mode, 1);
        assert_eq!(simulation.factorization_frequency, 200);
        assert_eq!(default.dual_pricing_mode, 3);
        assert_eq!(default.factorization_frequency, 0);
    }

    /// The CLP `BACKWARD` profile overrides only `dual_pricing_mode` (to `1`,
    /// full DSE) and `factorization_frequency` (to `200`), matching the default
    /// on every other field.
    #[test]
    fn clp_backward_profile_overrides_only_pricing_and_factorization() {
        let default = ClpProfile::default();
        let backward = <ClpProfile as PhaseProfiles>::BACKWARD;
        assert_ne!(backward, default);
        assert_eq!(backward.dual_pricing_mode, 1);
        assert_eq!(backward.factorization_frequency, 200);
        // Fields that are NOT overridden must match the default.
        assert_eq!(backward.perturbation, default.perturbation);
        assert_eq!(backward.scaling, default.scaling);
        assert_eq!(
            backward.primal_feasibility_tolerance,
            default.primal_feasibility_tolerance
        );
        assert_eq!(
            backward.dual_feasibility_tolerance,
            default.dual_feasibility_tolerance
        );
        assert_eq!(
            backward.simplex_iteration_limit,
            default.simplex_iteration_limit
        );
        assert_eq!(backward.algorithm, default.algorithm);
    }

    /// `Phase::profile()` returns the matching CLP per-phase profile for each
    /// variant (under `--features clp`, `ActiveProfile` resolves to `ClpProfile`).
    #[test]
    fn phase_profile_returns_matching_clp_profile() {
        assert_eq!(
            Phase::Forward.profile(),
            <ClpProfile as PhaseProfiles>::FORWARD
        );
        assert_eq!(
            Phase::Backward.profile(),
            <ClpProfile as PhaseProfiles>::BACKWARD
        );
        assert_eq!(
            Phase::Simulation.profile(),
            <ClpProfile as PhaseProfiles>::SIMULATION
        );
    }
}
