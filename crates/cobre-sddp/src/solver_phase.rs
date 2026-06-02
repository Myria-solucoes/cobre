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
use cobre_solver::{ClpAlgorithm, ClpProfile};
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
/// `FORWARD` and `SIMULATION` equal [`ClpProfile::default()`]'s field set.
/// `BACKWARD` is tuned for the backward pass: it pins full dual steepest-edge
/// pricing (`dual_pricing_mode = 1`) and sets an SDDP-tuned refactorization
/// cadence (`factorization_frequency = 200`), leaving every other field at the
/// default. This is the CLP-native analog of the `HiGHS` backward override of
/// `simplex_price_strategy`. These are CLP-native values, independently chosen
/// for CLP's own option surface — they are **not** a translation of the `HiGHS`
/// per-phase profiles. `algorithm` stays [`ClpAlgorithm::Dual`] in all three
/// phases (no phase selects the primal simplex). The full field literals are
/// written out because associated constants cannot use the
/// `..ClpProfile::default()` struct-update spread in const context.
#[cfg(feature = "clp")]
impl PhaseProfiles for ClpProfile {
    const FORWARD: Self = ClpProfile {
        perturbation: 102,
        scaling: 0,
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: cobre_solver::DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        algorithm: ClpAlgorithm::Dual,
        dual_pricing_mode: 3,       // CLP steepest-edge ctor default
        factorization_frequency: 0, // CLP internal default (sentinel)
    };
    const BACKWARD: Self = ClpProfile {
        perturbation: 102,
        scaling: 0,
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: cobre_solver::DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        algorithm: ClpAlgorithm::Dual,
        dual_pricing_mode: 1,         // full DSE pinned (backward-pass override)
        factorization_frequency: 200, // tuned refactor cadence (backward-pass override)
    };
    const SIMULATION: Self = ClpProfile {
        perturbation: 102,
        scaling: 0,
        primal_feasibility_tolerance: 1e-9,
        dual_feasibility_tolerance: 1e-9,
        simplex_iteration_limit: cobre_solver::DEFAULT_PROFILE_HEURISTIC_SENTINEL,
        algorithm: ClpAlgorithm::Dual,
        dual_pricing_mode: 3,       // CLP steepest-edge ctor default
        factorization_frequency: 0, // CLP internal default (sentinel)
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
        let base = match self {
            Phase::Forward => <cobre_solver::ActiveProfile as PhaseProfiles>::FORWARD,
            Phase::Backward => <cobre_solver::ActiveProfile as PhaseProfiles>::BACKWARD,
            Phase::Simulation => <cobre_solver::ActiveProfile as PhaseProfiles>::SIMULATION,
        };
        // Benchmark-only runtime overrides. Inert unless `COBRE_TUNE_*` is set,
        // so the production path returns the compile-time const profile above
        // unchanged and stays deterministic.
        tuning::apply_overrides(base)
    }
}

/// Benchmark-only solver-parameter overrides read from `COBRE_TUNE_*`
/// environment variables.
///
/// This is the runtime tuning seam for the local solver-parameter optimization
/// harness. It is **entirely inert unless an override variable is set**: with no
/// variables set, `apply_overrides` returns the compile-time per-phase profile
/// unchanged, so the default/production path and its determinism are untouched.
/// Overrides are *global* (applied to every phase) and parsed once on first use.
/// Recognized variables:
///
/// - `HiGHS`: `COBRE_TUNE_HIGHS_EDGE_WEIGHT` (i32), `_SCALE` (i32),
///   `_PRICE` (i32), `_PRIMAL_TOL` (f64), `_DUAL_TOL` (f64),
///   `_ITER_LIMIT` (u32). (`presolve` is not a profile field; it is tuned via
///   `COBRE_TUNE_HIGHS_PRESOLVE` inside `cobre-solver`.)
/// - `CLP`: `COBRE_TUNE_CLP_PERTURBATION` (i32), `_SCALING` (i32),
///   `_PRICING_MODE` (i32), `_FACTOR_FREQ` (i32), `_PRIMAL_TOL` (f64),
///   `_DUAL_TOL` (f64), `_ITER_LIMIT` (u32), `_ALGORITHM` (`dual`|`primal`).
mod tuning {
    use std::sync::OnceLock;

    /// Parses an integer env var; warns (once) and ignores a present-but-invalid
    /// value rather than silently running the baseline.
    fn env_i32(key: &str) -> Option<i32> {
        parse_env(key)
    }

    fn env_u32(key: &str) -> Option<u32> {
        parse_env(key)
    }

    fn env_f64(key: &str) -> Option<f64> {
        parse_env(key)
    }

    fn parse_env<T: std::str::FromStr>(key: &str) -> Option<T> {
        let raw = std::env::var(key).ok()?;
        if let Ok(v) = raw.trim().parse::<T>() {
            Some(v)
        } else {
            eprintln!(
                "[cobre tune] WARNING: {key}={raw:?} is not a valid {ty}; ignoring",
                ty = std::any::type_name::<T>()
            );
            None
        }
    }

    #[cfg(feature = "highs")]
    struct HighsOverrides {
        edge_weight: Option<i32>,
        scale: Option<i32>,
        price: Option<i32>,
        primal_tol: Option<f64>,
        dual_tol: Option<f64>,
        iter_limit: Option<u32>,
    }

    #[cfg(feature = "highs")]
    impl HighsOverrides {
        fn any(&self) -> bool {
            self.edge_weight.is_some()
                || self.scale.is_some()
                || self.price.is_some()
                || self.primal_tol.is_some()
                || self.dual_tol.is_some()
                || self.iter_limit.is_some()
        }
    }

    #[cfg(feature = "highs")]
    fn highs_overrides() -> &'static HighsOverrides {
        static OV: OnceLock<HighsOverrides> = OnceLock::new();
        OV.get_or_init(|| {
            let ov = HighsOverrides {
                edge_weight: env_i32("COBRE_TUNE_HIGHS_EDGE_WEIGHT"),
                scale: env_i32("COBRE_TUNE_HIGHS_SCALE"),
                price: env_i32("COBRE_TUNE_HIGHS_PRICE"),
                primal_tol: env_f64("COBRE_TUNE_HIGHS_PRIMAL_TOL"),
                dual_tol: env_f64("COBRE_TUNE_HIGHS_DUAL_TOL"),
                iter_limit: env_u32("COBRE_TUNE_HIGHS_ITER_LIMIT"),
            };
            if ov.any() {
                eprintln!(
                    "[cobre tune] HiGHS profile overrides active: \
                     edge_weight={:?} scale={:?} price={:?} primal_tol={:?} \
                     dual_tol={:?} iter_limit={:?}",
                    ov.edge_weight, ov.scale, ov.price, ov.primal_tol, ov.dual_tol, ov.iter_limit
                );
            }
            ov
        })
    }

    /// Applies the active `HiGHS` overrides to a per-phase profile.
    #[cfg(feature = "highs")]
    pub(super) fn apply_overrides(mut p: cobre_solver::HighsProfile) -> cobre_solver::HighsProfile {
        let ov = highs_overrides();
        if let Some(v) = ov.edge_weight {
            p.simplex_dual_edge_weight_strategy = v;
        }
        if let Some(v) = ov.scale {
            p.simplex_scale_strategy = v;
        }
        if let Some(v) = ov.price {
            p.simplex_price_strategy = v;
        }
        if let Some(v) = ov.primal_tol {
            p.primal_feasibility_tolerance = v;
        }
        if let Some(v) = ov.dual_tol {
            p.dual_feasibility_tolerance = v;
        }
        if let Some(v) = ov.iter_limit {
            p.simplex_iteration_limit = v;
        }
        p
    }

    #[cfg(feature = "clp")]
    struct ClpOverrides {
        perturbation: Option<i32>,
        scaling: Option<i32>,
        pricing_mode: Option<i32>,
        factor_freq: Option<i32>,
        primal_tol: Option<f64>,
        dual_tol: Option<f64>,
        iter_limit: Option<u32>,
        algorithm: Option<cobre_solver::ClpAlgorithm>,
    }

    #[cfg(feature = "clp")]
    impl ClpOverrides {
        fn any(&self) -> bool {
            self.perturbation.is_some()
                || self.scaling.is_some()
                || self.pricing_mode.is_some()
                || self.factor_freq.is_some()
                || self.primal_tol.is_some()
                || self.dual_tol.is_some()
                || self.iter_limit.is_some()
                || self.algorithm.is_some()
        }
    }

    #[cfg(feature = "clp")]
    fn env_clp_algorithm(key: &str) -> Option<cobre_solver::ClpAlgorithm> {
        let raw = std::env::var(key).ok()?;
        match raw.trim().to_ascii_lowercase().as_str() {
            "dual" => Some(cobre_solver::ClpAlgorithm::Dual),
            "primal" => Some(cobre_solver::ClpAlgorithm::Primal),
            other => {
                eprintln!("[cobre tune] WARNING: {key}={other:?} is not dual|primal; ignoring");
                None
            }
        }
    }

    #[cfg(feature = "clp")]
    fn clp_overrides() -> &'static ClpOverrides {
        static OV: OnceLock<ClpOverrides> = OnceLock::new();
        OV.get_or_init(|| {
            let ov = ClpOverrides {
                perturbation: env_i32("COBRE_TUNE_CLP_PERTURBATION"),
                scaling: env_i32("COBRE_TUNE_CLP_SCALING"),
                pricing_mode: env_i32("COBRE_TUNE_CLP_PRICING_MODE"),
                factor_freq: env_i32("COBRE_TUNE_CLP_FACTOR_FREQ"),
                primal_tol: env_f64("COBRE_TUNE_CLP_PRIMAL_TOL"),
                dual_tol: env_f64("COBRE_TUNE_CLP_DUAL_TOL"),
                iter_limit: env_u32("COBRE_TUNE_CLP_ITER_LIMIT"),
                algorithm: env_clp_algorithm("COBRE_TUNE_CLP_ALGORITHM"),
            };
            if ov.any() {
                eprintln!(
                    "[cobre tune] CLP profile overrides active: \
                     perturbation={:?} scaling={:?} pricing_mode={:?} \
                     factor_freq={:?} primal_tol={:?} dual_tol={:?} \
                     iter_limit={:?} algorithm={:?}",
                    ov.perturbation,
                    ov.scaling,
                    ov.pricing_mode,
                    ov.factor_freq,
                    ov.primal_tol,
                    ov.dual_tol,
                    ov.iter_limit,
                    ov.algorithm
                );
            }
            ov
        })
    }

    /// Applies the active `CLP` overrides to a per-phase profile.
    #[cfg(feature = "clp")]
    pub(super) fn apply_overrides(mut p: cobre_solver::ClpProfile) -> cobre_solver::ClpProfile {
        let ov = clp_overrides();
        if let Some(v) = ov.perturbation {
            p.perturbation = v;
        }
        if let Some(v) = ov.scaling {
            p.scaling = v;
        }
        if let Some(v) = ov.pricing_mode {
            p.dual_pricing_mode = v;
        }
        if let Some(v) = ov.factor_freq {
            p.factorization_frequency = v;
        }
        if let Some(v) = ov.primal_tol {
            p.primal_feasibility_tolerance = v;
        }
        if let Some(v) = ov.dual_tol {
            p.dual_feasibility_tolerance = v;
        }
        if let Some(v) = ov.iter_limit {
            p.simplex_iteration_limit = v;
        }
        if let Some(v) = ov.algorithm {
            p.algorithm = v;
        }
        p
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

    /// The CLP `FORWARD`/`SIMULATION` per-phase profiles equal
    /// [`ClpProfile::default()`].
    #[test]
    fn clp_phase_profiles_forward_simulation_equal_default() {
        let default = ClpProfile::default();
        assert_eq!(<ClpProfile as PhaseProfiles>::FORWARD, default);
        assert_eq!(<ClpProfile as PhaseProfiles>::SIMULATION, default);
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
