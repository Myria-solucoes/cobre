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

use std::num::NonZeroUsize;

use cobre_io::config::{BackwardScheduler, PhaseSolverProfileConfig};
#[cfg(feature = "highs")]
use cobre_io::config::{DualEdgeWeight, PriceStrategy, ScaleStrategy};
#[cfg(feature = "highs")]
use cobre_solver::HighsProfile;
use cobre_solver::{ActiveProfile, DEFAULT_PROFILE_HEURISTIC_SENTINEL};
#[cfg(feature = "clp")]
use cobre_solver::{ClpAlgorithm, ClpProfile};

use crate::SddpError;

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

/// Resolved per-run backward/forward tuning, threaded from [`crate::setup::StudySetup`]
/// into `TrainingSession::new` once per training run: the solver profiles plus
/// the backward-pass scheduler (`training.backward_scheduler`/
/// `training.opening_block_size`), which has no forward-pass analogue but
/// travels the same seam.
///
/// `Default` reproduces [`Phase::profile`]'s byte-neutral constants and the
/// byte-neutral trial-point scheduler, so a caller with no config override can
/// pass `SolverProfiles::default()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverProfiles {
    /// Resolved forward-pass profile.
    pub forward: ActiveProfile,
    /// Resolved backward-pass profile.
    pub backward: ActiveProfile,
    /// Backward-pass scheduler (`training.backward_scheduler`).
    pub backward_scheduler: BackwardScheduler,
    /// Opening-block size override for `backward_scheduler = opening_block`
    /// (`training.opening_block_size`).
    pub opening_block_size: Option<NonZeroUsize>,
}

impl Default for SolverProfiles {
    fn default() -> Self {
        Self {
            forward: Phase::Forward.profile(),
            backward: Phase::Backward.profile(),
            backward_scheduler: BackwardScheduler::default(),
            opening_block_size: None,
        }
    }
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

    /// Resolves this phase's [`ActiveProfile`], layered as base constant
    /// ([`Self::profile`]) + named preset + per-field overrides (a later layer
    /// wins). `config` absent returns [`Self::profile`] unchanged — the
    /// byte-neutral default. An unrecognized preset name falls back to the
    /// base constant; call `validate_phase_solver_config` on the same
    /// `config` to reject it instead.
    #[must_use]
    pub fn resolve_profile(self, config: Option<&PhaseSolverProfileConfig>) -> ActiveProfile {
        resolve_active_profile(self.profile(), config)
    }
}

fn phase_config_key(phase: Phase) -> &'static str {
    match phase {
        Phase::Forward => "forward",
        Phase::Backward => "backward",
        Phase::Simulation => "simulation",
    }
}

/// Validates a resolved per-phase solver config against the compiled
/// backend's support matrix (decision D11: everything HiGHS-scoped for v1).
///
/// `config` absent is always `Ok(())` — the byte-neutral path
/// ([`Phase::resolve_profile`]) is untouched by this validator.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] naming the offending preset or field,
/// its value, and the compiled backend when `config` sets something the
/// compiled backend does not support.
pub(crate) fn validate_phase_solver_config(
    config: Option<&PhaseSolverProfileConfig>,
    phase: Phase,
) -> Result<(), SddpError> {
    match config {
        Some(config) => validate_backend_support(config, phase),
        None => Ok(()),
    }
}

/// Under `highs`, every [`DualEdgeWeight`]/[`ScaleStrategy`]/[`PriceStrategy`]
/// variant is supported (the enums are already closed to the supported set),
/// so only the preset name and the tolerance's finiteness need checking.
#[cfg(feature = "highs")]
fn validate_backend_support(
    config: &PhaseSolverProfileConfig,
    phase: Phase,
) -> Result<(), SddpError> {
    let phase_key = phase_config_key(phase);
    if let Some(preset) = config.preset.as_deref()
        && preset != "backward_tuned_v1"
    {
        return Err(SddpError::Validation(format!(
            "unknown solver profile preset {preset:?} for phase \"{phase_key}\"; known presets: \"backward_tuned_v1\""
        )));
    }
    if let Some(tolerance) = config.primal_feasibility_tolerance
        && !tolerance.is_finite()
    {
        return Err(SddpError::Validation(format!(
            "unsupported primal_feasibility_tolerance {tolerance} for backend \"highs\" in phase \"{phase_key}\": value must be finite"
        )));
    }
    Ok(())
}

// CLP support is minimal for v1 (decision D11): reject every preset/override
// instead of silently applying a HiGHS-flavored value to CLP's own option
// surface.
#[cfg(feature = "clp")]
fn validate_backend_support(
    config: &PhaseSolverProfileConfig,
    phase: Phase,
) -> Result<(), SddpError> {
    let phase_key = phase_config_key(phase);
    if let Some(preset) = config.preset.as_deref() {
        return Err(clp_unsupported(format!("preset {preset:?}"), phase_key));
    }
    if config.dual_edge_weight.is_some() {
        return Err(clp_unsupported("field \"dual_edge_weight\"", phase_key));
    }
    if config.scale.is_some() {
        return Err(clp_unsupported("field \"scale\"", phase_key));
    }
    if config.price.is_some() {
        return Err(clp_unsupported("field \"price\"", phase_key));
    }
    if let Some(tolerance) = config.primal_feasibility_tolerance {
        return Err(clp_unsupported(
            format!("field \"primal_feasibility_tolerance\" ({tolerance})"),
            phase_key,
        ));
    }
    Ok(())
}

#[cfg(feature = "clp")]
fn clp_unsupported(item: impl std::fmt::Display, phase_key: &str) -> SddpError {
    SddpError::Validation(format!(
        "solver profile {item} for phase \"{phase_key}\" is unsupported on backend \"clp\": rejected until CLP is re-measured"
    ))
}

/// The `backward_tuned_v1` config preset — the measured tuned bundle (dual
/// `SteepestEdge`, Curtis–Reid scaling, `Row` pricing, loosened primal
/// feasibility tolerance) selected via
/// `training.solver.backward = {"preset": "backward_tuned_v1"}`. Fields the
/// bundle does not tune keep [`BACKWARD_PROFILE`]'s values.
#[cfg(feature = "highs")]
pub const BACKWARD_TUNED_V1_PRESET: HighsProfile = HighsProfile {
    primal_feasibility_tolerance: 1e-7,
    dual_feasibility_tolerance: 1e-9,
    simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
    ipm_iteration_limit: 10_000,
    simplex_dual_edge_weight_strategy: 2, // SteepestEdge
    simplex_scale_strategy: 2,            // Curtis–Reid
    simplex_price_strategy: 1,            // Row
};

#[cfg(feature = "highs")]
fn highs_dual_edge_weight(value: DualEdgeWeight) -> i32 {
    match value {
        DualEdgeWeight::Dantzig => 0,
        DualEdgeWeight::Devex => 1,
        DualEdgeWeight::SteepestEdge => 2,
    }
}

#[cfg(feature = "highs")]
fn highs_scale_strategy(value: ScaleStrategy) -> i32 {
    match value {
        ScaleStrategy::Off => 0,
        ScaleStrategy::SolverScaling => 2, // Curtis–Reid
    }
}

#[cfg(feature = "highs")]
fn highs_price_strategy(value: PriceStrategy) -> i32 {
    match value {
        PriceStrategy::Row => 1,
        PriceStrategy::RowHyperSparse => 2,
    }
}

/// Layers `config` onto `base`: preset replaces `base` wholesale (an
/// unrecognized name is silently ignored — validation is a separate step),
/// then each `Some` per-field override replaces the corresponding option on
/// top of whichever profile the preset step produced.
#[cfg(feature = "highs")]
fn resolve_active_profile(
    base: HighsProfile,
    config: Option<&PhaseSolverProfileConfig>,
) -> HighsProfile {
    let Some(config) = config else {
        return base;
    };
    let mut profile = match config.preset.as_deref() {
        Some("backward_tuned_v1") => BACKWARD_TUNED_V1_PRESET,
        _ => base,
    };
    if let Some(weight) = config.dual_edge_weight {
        profile.simplex_dual_edge_weight_strategy = highs_dual_edge_weight(weight);
    }
    if let Some(scale) = config.scale {
        profile.simplex_scale_strategy = highs_scale_strategy(scale);
    }
    if let Some(price) = config.price {
        profile.simplex_price_strategy = highs_price_strategy(price);
    }
    if let Some(tolerance) = config.primal_feasibility_tolerance {
        profile.primal_feasibility_tolerance = tolerance;
    }
    profile
}

// CLP support is minimal for v1 (decision D11): every override is deferred to
// the compiled-backend validation step, which hard-rejects a CLP config
// instead of silently applying a HiGHS-flavored preset/override to CLP's own
// option surface.
#[cfg(feature = "clp")]
fn resolve_active_profile(
    base: ClpProfile,
    _config: Option<&PhaseSolverProfileConfig>,
) -> ClpProfile {
    base
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

    assert!(BACKWARD_TUNED_V1_PRESET.primal_feasibility_tolerance == 1e-7);
    assert!(BACKWARD_TUNED_V1_PRESET.dual_feasibility_tolerance == 1e-9);
    assert!(BACKWARD_TUNED_V1_PRESET.simplex_iteration_limit == DEFAULT_PROFILE_HEURISTIC_SENTINEL);
    assert!(BACKWARD_TUNED_V1_PRESET.ipm_iteration_limit == 10_000);
    assert!(BACKWARD_TUNED_V1_PRESET.simplex_dual_edge_weight_strategy == 2);
    assert!(BACKWARD_TUNED_V1_PRESET.simplex_scale_strategy == 2);
    assert!(BACKWARD_TUNED_V1_PRESET.simplex_price_strategy == 1);
};

#[cfg(test)]
mod validate_phase_solver_config_tests {
    use super::{Phase, validate_phase_solver_config};

    /// A fully-absent config is always accepted, regardless of the compiled
    /// backend — the byte-neutral resolution path stays untouched.
    #[test]
    fn absent_config_is_accepted() {
        assert!(validate_phase_solver_config(None, Phase::Forward).is_ok());
        assert!(validate_phase_solver_config(None, Phase::Backward).is_ok());
        assert!(validate_phase_solver_config(None, Phase::Simulation).is_ok());
    }
}

#[cfg(all(test, feature = "highs"))]
mod highs_tests {
    use cobre_io::config::{DualEdgeWeight, PhaseSolverProfileConfig, PriceStrategy};
    use cobre_solver::HighsProfile;

    use super::{
        BACKWARD_PROFILE, BACKWARD_TUNED_V1_PRESET, FORWARD_PROFILE, Phase, PhaseProfiles,
        SIMULATION_PROFILE, validate_phase_solver_config,
    };

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

    // ── `resolve_profile` ────────────────────────────────────────────────────

    /// Absent config resolves to the current per-phase constant, unchanged —
    /// the byte-neutrality gate.
    #[test]
    fn resolve_profile_absent_config_equals_named_constant() {
        assert_eq!(Phase::Forward.resolve_profile(None), FORWARD_PROFILE);
        assert_eq!(Phase::Backward.resolve_profile(None), BACKWARD_PROFILE);
        assert_eq!(Phase::Simulation.resolve_profile(None), SIMULATION_PROFILE);
    }

    /// The `backward_tuned_v1` preset alone resolves to the exact measured
    /// int/tolerance bundle.
    #[test]
    fn resolve_profile_preset_resolves_to_exact_bundle() {
        let config = PhaseSolverProfileConfig {
            preset: Some("backward_tuned_v1".to_string()),
            dual_edge_weight: None,
            scale: None,
            price: None,
            primal_feasibility_tolerance: None,
        };
        let resolved = Phase::Backward.resolve_profile(Some(&config));
        assert_eq!(resolved, BACKWARD_TUNED_V1_PRESET);
        assert_eq!(resolved.simplex_dual_edge_weight_strategy, 2);
        assert_eq!(resolved.simplex_scale_strategy, 2);
        assert_eq!(resolved.simplex_price_strategy, 1);
        assert_eq!(resolved.primal_feasibility_tolerance, 1e-7);
    }

    /// A per-field override beats the preset it is layered on top of; fields
    /// the override does not touch keep the preset's values.
    #[test]
    fn resolve_profile_per_field_override_beats_preset() {
        let config = PhaseSolverProfileConfig {
            preset: Some("backward_tuned_v1".to_string()),
            dual_edge_weight: None,
            scale: None,
            price: Some(PriceStrategy::RowHyperSparse),
            primal_feasibility_tolerance: None,
        };
        let resolved = Phase::Backward.resolve_profile(Some(&config));
        assert_eq!(resolved.simplex_price_strategy, 2);
        assert_eq!(resolved.simplex_dual_edge_weight_strategy, 2);
        assert_eq!(resolved.simplex_scale_strategy, 2);
        assert_eq!(resolved.primal_feasibility_tolerance, 1e-7);
    }

    /// An unrecognized preset name falls back to the base constant
    /// (`validate_phase_solver_config` rejects it before resolution); it must
    /// not panic or silently corrupt other fields.
    #[test]
    fn resolve_profile_unknown_preset_falls_back_to_base_constant() {
        let config = PhaseSolverProfileConfig {
            preset: Some("turbo".to_string()),
            dual_edge_weight: None,
            scale: None,
            price: None,
            primal_feasibility_tolerance: None,
        };
        assert_eq!(
            Phase::Backward.resolve_profile(Some(&config)),
            BACKWARD_PROFILE
        );
    }

    /// Per-field overrides apply directly on the base constant when no preset
    /// is named.
    #[test]
    fn resolve_profile_field_overrides_without_preset_apply_on_base_constant() {
        let config = PhaseSolverProfileConfig {
            preset: None,
            dual_edge_weight: Some(DualEdgeWeight::Dantzig),
            scale: None,
            price: None,
            primal_feasibility_tolerance: None,
        };
        let resolved = Phase::Forward.resolve_profile(Some(&config));
        assert_eq!(resolved.simplex_dual_edge_weight_strategy, 0);
        assert_eq!(
            resolved.simplex_scale_strategy,
            FORWARD_PROFILE.simplex_scale_strategy
        );
        assert_eq!(
            resolved.simplex_price_strategy,
            FORWARD_PROFILE.simplex_price_strategy
        );
        assert_eq!(
            resolved.primal_feasibility_tolerance,
            FORWARD_PROFILE.primal_feasibility_tolerance
        );
    }

    // ── `validate_phase_solver_config` (HiGHS) ──────────────────────────────

    /// An unrecognized preset name is rejected, naming the preset and the phase.
    #[test]
    fn validate_phase_solver_config_rejects_unknown_preset() {
        let config = PhaseSolverProfileConfig {
            preset: Some("turbo".to_string()),
            dual_edge_weight: None,
            scale: None,
            price: None,
            primal_feasibility_tolerance: None,
        };
        let err = validate_phase_solver_config(Some(&config), Phase::Backward)
            .expect_err("unknown preset must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("turbo"), "message must name the preset: {msg}");
        assert!(
            msg.contains("backward"),
            "message must name the phase: {msg}"
        );
    }

    /// A non-finite `primal_feasibility_tolerance` is rejected, naming the
    /// field, its value, and the backend.
    #[test]
    fn validate_phase_solver_config_rejects_non_finite_tolerance() {
        let config = PhaseSolverProfileConfig {
            preset: None,
            dual_edge_weight: None,
            scale: None,
            price: None,
            primal_feasibility_tolerance: Some(f64::NAN),
        };
        let err = validate_phase_solver_config(Some(&config), Phase::Forward)
            .expect_err("non-finite tolerance must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("primal_feasibility_tolerance"),
            "message must name the field: {msg}"
        );
        assert!(
            msg.contains("highs"),
            "message must name the backend: {msg}"
        );
    }

    /// The known preset plus a per-field override the closed enums allow are
    /// accepted without error.
    #[test]
    fn validate_phase_solver_config_accepts_known_preset_and_overrides() {
        let config = PhaseSolverProfileConfig {
            preset: Some("backward_tuned_v1".to_string()),
            dual_edge_weight: Some(DualEdgeWeight::Dantzig),
            scale: None,
            price: Some(PriceStrategy::RowHyperSparse),
            primal_feasibility_tolerance: Some(1e-8),
        };
        assert!(validate_phase_solver_config(Some(&config), Phase::Backward).is_ok());
    }
}

#[cfg(all(test, feature = "clp"))]
mod clp_tests {
    use cobre_io::config::{DualEdgeWeight, PhaseSolverProfileConfig};
    use cobre_solver::{ClpAlgorithm, ClpProfile};

    use super::{Phase, PhaseProfiles, validate_phase_solver_config};

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

    /// CLP support is minimal for v1 (D11): `resolve_profile` returns the base
    /// constant regardless of `config` — absent config is byte-neutral, and a
    /// present one is silently ignored pending the compiled-backend reject.
    #[test]
    fn resolve_profile_ignores_config_under_clp() {
        assert_eq!(
            Phase::Backward.resolve_profile(None),
            Phase::Backward.profile()
        );
        let config = PhaseSolverProfileConfig {
            preset: Some("backward_tuned_v1".to_string()),
            dual_edge_weight: None,
            scale: None,
            price: None,
            primal_feasibility_tolerance: None,
        };
        assert_eq!(
            Phase::Backward.resolve_profile(Some(&config)),
            Phase::Backward.profile()
        );
    }

    // ── `validate_phase_solver_config` (CLP, D11 hard-reject) ───────────────

    /// The tuned preset is hard-rejected on CLP, naming CLP and the preset.
    #[test]
    fn validate_phase_solver_config_rejects_clp_preset() {
        let config = PhaseSolverProfileConfig {
            preset: Some("backward_tuned_v1".to_string()),
            dual_edge_weight: None,
            scale: None,
            price: None,
            primal_feasibility_tolerance: None,
        };
        let err = validate_phase_solver_config(Some(&config), Phase::Backward)
            .expect_err("CLP must hard-reject the tuned preset");
        let msg = err.to_string();
        assert!(msg.contains("clp"), "message must name the backend: {msg}");
        assert!(
            msg.contains("backward_tuned_v1"),
            "message must name the preset: {msg}"
        );
    }

    /// A per-field override with no preset is also hard-rejected on CLP — D11
    /// rejects every HiGHS-only override, not only the preset.
    #[test]
    fn validate_phase_solver_config_rejects_clp_override_without_preset() {
        let config = PhaseSolverProfileConfig {
            preset: None,
            dual_edge_weight: Some(DualEdgeWeight::Dantzig),
            scale: None,
            price: None,
            primal_feasibility_tolerance: None,
        };
        let err = validate_phase_solver_config(Some(&config), Phase::Forward)
            .expect_err("CLP must hard-reject an unsupported override");
        assert!(err.to_string().contains("clp"));
    }
}
