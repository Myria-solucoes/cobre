//! SDDP algorithm phase enum, the backend-agnostic [`PhaseProfiles`] trait, and
//! the per-phase named solver-profile constants.
//!
//! Each [`Phase`] variant maps to a backend profile that configures the LP
//! solver for that phase. For the `HiGHS` backend, [`FORWARD_PROFILE`] and
//! [`SIMULATION_PROFILE`] match [`HighsProfile::default()`] field-for-field.
//! [`BACKWARD_PROFILE`] overrides one option — `simplex_price_strategy`
//! (`RowHyperSparse`, value `2`) — to exploit sparse cut subgradient rows
//! that dominate the backward LP row count. Edge-weight strategy stays at
//! Devex (the [`HighsProfile::default`] value) after empirical sweeps
//! showed Dantzig and `SteepestEdge` alternatives net worse on wall time
//! and tail latency respectively.
//!
//! Compile-time assertions at the bottom of this module catch any future drift
//! between the named `HiGHS` constants and the documented field values.
//!
//! Every field of both `HighsProfile` and `ClpProfile` is a primitive
//! `f64`/`u32`/`i32`, so both profile structs are fully const-constructible and
//! the [`PhaseProfiles`] trait exposes its per-phase profiles as **associated
//! constants** (`const FORWARD`/`BACKWARD`/`SIMULATION`) rather than methods.
//!
//! ## Why the per-phase profiles live here, not in `cobre-solver`
//!
//! The per-phase profiles are authored here as concrete backend values, which
//! couples this module to whichever backend is active. This is deliberate, not
//! an oversight: the *mapping* "which phase wants which solver behaviour" (e.g.
//! the backward pass exploits sparse cut-subgradient rows) is algorithm
//! knowledge that must live in the algorithm crate, while `cobre-solver` is kept
//! strictly backend-agnostic and so cannot know about SDDP phases. The
//! [`PhaseProfiles`] trait abstracts only *which phase is running* — never the
//! tuning content — and its impls keep the SDDP-tuned values inside
//! `cobre-sddp`. The trait is local to this crate and the implemented profile
//! types are foreign, which the orphan rule permits.

#[cfg(feature = "clp")]
use cobre_solver::ClpProfile;
#[cfg(feature = "highs")]
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

/// Per-phase identity selection of a backend solver profile.
///
/// `PhaseProfiles` is a backend-agnostic vocabulary that abstracts only *which
/// phase is running* — `FORWARD`, `BACKWARD`, or `SIMULATION` — never the tuning
/// content. Each implementing backend profile type supplies its own per-phase
/// values: the `HiGHS` impl reproduces the SDDP-tuned `HighsProfile` constants
/// in this module bit-for-bit, while the CLP impl supplies CLP-native values.
///
/// The trait is defined in `cobre-sddp` (the algorithm crate), not in
/// `cobre-solver`, so the SDDP-tuned per-phase *values* stay with the algorithm
/// and `cobre-solver` remains strictly backend-generic. The trait is local and
/// the implemented profile types are foreign, which the orphan rule permits.
///
/// The three members are **associated constants** because every field of every
/// backend profile struct is const-constructible (see the module docs).
pub trait PhaseProfiles: Sized {
    /// Profile applied when entering the forward pass.
    const FORWARD: Self;
    /// Profile applied when entering the backward pass.
    const BACKWARD: Self;
    /// Profile applied when entering policy simulation.
    const SIMULATION: Self;
}

/// Per-phase `HighsProfile` selection.
///
/// Reproduces the SDDP-tuned [`FORWARD_PROFILE`]/[`BACKWARD_PROFILE`]/
/// [`SIMULATION_PROFILE`] constants of this module bit-for-bit:
/// `FORWARD` and `SIMULATION` equal [`HighsProfile::default()`]; `BACKWARD`
/// overrides only `simplex_price_strategy` to `2` (`RowHyperSparse`). The full
/// field literals are written out because associated constants cannot use the
/// `..HighsProfile::default()` struct-update spread in const context.
#[cfg(feature = "highs")]
impl PhaseProfiles for HighsProfile {
    const FORWARD: Self = HighsProfile {
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        ipm_iteration_limit: 10_000,
        simplex_dual_edge_weight_strategy: 1, // Devex (default)
        simplex_scale_strategy: 0, // Off (default; cobre prescaler conditions the matrix)
        simplex_price_strategy: 1, // Row (default)
    };
    const BACKWARD: Self = HighsProfile {
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        ipm_iteration_limit: 10_000,
        simplex_dual_edge_weight_strategy: 1, // Devex (matches default)
        simplex_scale_strategy: 0, // Off (matches default; cobre prescaler conditions the matrix)
        simplex_price_strategy: 2, // RowHyperSparse override
    };
    const SIMULATION: Self = HighsProfile {
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        ipm_iteration_limit: 10_000,
        simplex_dual_edge_weight_strategy: 1, // Devex (default)
        simplex_scale_strategy: 0, // Off (default; cobre prescaler conditions the matrix)
        simplex_price_strategy: 1, // Row (default)
    };
}

/// Per-phase `ClpProfile` selection.
///
/// All three phases use [`ClpProfile::default()`]'s field set. These are
/// CLP-native values, independently chosen for CLP's own option surface — they
/// are **not** a translation of the `HiGHS` per-phase profiles. CLP has no
/// per-phase override yet; a future phase may differentiate `BACKWARD`.
#[cfg(feature = "clp")]
impl PhaseProfiles for ClpProfile {
    const FORWARD: Self = ClpProfile {
        perturbation: 102,
        scaling: 0,
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: cobre_solver::DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    };
    const BACKWARD: Self = ClpProfile {
        perturbation: 102,
        scaling: 0,
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: cobre_solver::DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    };
    const SIMULATION: Self = ClpProfile {
        perturbation: 102,
        scaling: 0,
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: cobre_solver::DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    };
}

/// Solver profile applied during the SDDP forward pass.
///
/// All fields equal [`HighsProfile::default()`], which uses the tightened
/// `1e-9` primal/dual feasibility tolerances.
#[cfg(feature = "highs")]
pub const FORWARD_PROFILE: HighsProfile = HighsProfile {
    primal_feasibility_tolerance: 1e-9,
    dual_feasibility_tolerance: 1e-9,
    simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    ipm_iteration_limit: 10_000,
    simplex_dual_edge_weight_strategy: 1, // Devex (default)
    simplex_scale_strategy: 0,            // Off (default; cobre prescaler conditions the matrix)
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
#[cfg(feature = "highs")]
pub const BACKWARD_PROFILE: HighsProfile = HighsProfile {
    primal_feasibility_tolerance: 1e-9,
    dual_feasibility_tolerance: 1e-9,
    simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    ipm_iteration_limit: 10_000,
    simplex_dual_edge_weight_strategy: 1, // Devex (matches default)
    simplex_scale_strategy: 0, // Off (matches default; cobre prescaler conditions the matrix)
    simplex_price_strategy: 2, // RowHyperSparse override
};

/// Solver profile applied during policy simulation.
///
/// All fields equal [`HighsProfile::default()`], which uses the tightened
/// `1e-9` primal/dual feasibility tolerances.
#[cfg(feature = "highs")]
pub const SIMULATION_PROFILE: HighsProfile = HighsProfile {
    primal_feasibility_tolerance: 1e-9,
    dual_feasibility_tolerance: 1e-9,
    simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    ipm_iteration_limit: 10_000,
    simplex_dual_edge_weight_strategy: 1, // Devex (default)
    simplex_scale_strategy: 0,            // Off (default; cobre prescaler conditions the matrix)
    simplex_price_strategy: 1,            // Row (default)
};

impl Phase {
    /// Returns the [`cobre_solver::ActiveProfile`] that should be applied when
    /// entering this phase.
    ///
    /// `ActiveProfile` is the compile-time-selected backend profile
    /// (`HighsProfile` under `--features highs`, `ClpProfile` under
    /// `--features clp`). The per-phase value is delegated to the backend's
    /// [`PhaseProfiles`] impl, so under `HiGHS` the returned values are
    /// bit-for-bit identical to the `FORWARD_PROFILE`/`BACKWARD_PROFILE`/
    /// `SIMULATION_PROFILE` named constants.
    ///
    /// The returned profile is cheap to copy (`ActiveProfile` is `Copy` for both
    /// backends) and should be passed to `ProfiledSolver::set_profile` at phase
    /// entry.
    #[must_use]
    pub fn profile(self) -> cobre_solver::ActiveProfile {
        match self {
            Phase::Forward => <cobre_solver::ActiveProfile as PhaseProfiles>::FORWARD,
            Phase::Backward => <cobre_solver::ActiveProfile as PhaseProfiles>::BACKWARD,
            Phase::Simulation => <cobre_solver::ActiveProfile as PhaseProfiles>::SIMULATION,
        }
    }
}

// ── Compile-time drift guards ──────────────────────────────────────────────
//
// These assertions run at compile time and catch any future divergence between
// the named profile constants and their documented field values. If a field is
// added to `HighsProfile`, the compiler will reject any const struct literal
// that does not include it, forcing an explicit update here.

#[cfg(feature = "highs")]
const _: () = {
    // FORWARD_PROFILE — all fields equal HighsProfile::default()
    assert!(FORWARD_PROFILE.primal_feasibility_tolerance == 1e-9);
    assert!(FORWARD_PROFILE.dual_feasibility_tolerance == 1e-9);
    assert!(FORWARD_PROFILE.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL);
    assert!(FORWARD_PROFILE.ipm_iteration_limit == 10_000);
    assert!(FORWARD_PROFILE.simplex_dual_edge_weight_strategy == 1);
    assert!(FORWARD_PROFILE.simplex_scale_strategy == 0);
    assert!(FORWARD_PROFILE.simplex_price_strategy == 1);

    // BACKWARD_PROFILE — price=2 (RowHyperSparse), other fields equal default
    assert!(BACKWARD_PROFILE.primal_feasibility_tolerance == 1e-9);
    assert!(BACKWARD_PROFILE.dual_feasibility_tolerance == 1e-9);
    assert!(BACKWARD_PROFILE.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL);
    assert!(BACKWARD_PROFILE.ipm_iteration_limit == 10_000);
    assert!(BACKWARD_PROFILE.simplex_dual_edge_weight_strategy == 1);
    assert!(BACKWARD_PROFILE.simplex_scale_strategy == 0);
    assert!(BACKWARD_PROFILE.simplex_price_strategy == 2);

    // SIMULATION_PROFILE — all fields equal HighsProfile::default()
    assert!(SIMULATION_PROFILE.primal_feasibility_tolerance == 1e-9);
    assert!(SIMULATION_PROFILE.dual_feasibility_tolerance == 1e-9);
    assert!(SIMULATION_PROFILE.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL);
    assert!(SIMULATION_PROFILE.ipm_iteration_limit == 10_000);
    assert!(SIMULATION_PROFILE.simplex_dual_edge_weight_strategy == 1);
    assert!(SIMULATION_PROFILE.simplex_scale_strategy == 0);
    assert!(SIMULATION_PROFILE.simplex_price_strategy == 1);

    // PhaseProfiles for HighsProfile — bit-for-bit identical to the named
    // constants above, proving numeric inertness of the trait abstraction.
    assert!(matches!(
        <HighsProfile as PhaseProfiles>::FORWARD.simplex_price_strategy,
        1
    ));
    assert!(matches!(
        <HighsProfile as PhaseProfiles>::BACKWARD.simplex_price_strategy,
        2
    ));
    assert!(matches!(
        <HighsProfile as PhaseProfiles>::SIMULATION.simplex_price_strategy,
        1
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

    /// Forward and simulation named constants equal [`HighsProfile::default()`].
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

    /// The `PhaseProfiles` impl's `FORWARD`/`SIMULATION` equal the default.
    #[test]
    fn phase_profiles_forward_simulation_equal_default() {
        let default = HighsProfile::default();
        assert_eq!(<HighsProfile as PhaseProfiles>::FORWARD, default);
        assert_eq!(<HighsProfile as PhaseProfiles>::SIMULATION, default);
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
    use cobre_solver::ClpProfile;

    use super::{Phase, PhaseProfiles};

    /// All three CLP per-phase profiles equal [`ClpProfile::default()`].
    #[test]
    fn clp_phase_profiles_all_equal_default() {
        let default = ClpProfile::default();
        assert_eq!(<ClpProfile as PhaseProfiles>::FORWARD, default);
        assert_eq!(<ClpProfile as PhaseProfiles>::BACKWARD, default);
        assert_eq!(<ClpProfile as PhaseProfiles>::SIMULATION, default);
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
