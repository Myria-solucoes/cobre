//! CLP simplex-algorithm selection and tuning profile.
//!
//! Leaf submodule: owns the [`ClpAlgorithm`] selector and the [`ClpProfile`]
//! value type plus its [`Default`] tuning surface. `solver`, `retry`, and
//! `interface` read these via `super::config::{…}`.

use crate::DEFAULT_PROFILE_HEURISTIC_SENTINEL;

/// Simplex algorithm selection for [`ClpProfile`].
///
/// Selects which simplex algorithm `solve` runs: the **dual** simplex
/// (`Clp_dual`) or the **primal** simplex (`Clp_primal`). The dual simplex
/// re-optimizes efficiently after bound changes (a dual-feasible basis stays
/// dual-feasible), while the primal simplex is advantageous when a
/// primal-feasible starting basis is available. Both reach the same optimum;
/// only the iteration path differs. The default is [`ClpAlgorithm::Dual`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClpAlgorithm {
    /// Dual simplex (`Clp_dual`).
    #[default]
    Dual,
    /// Primal simplex (`Clp_primal`).
    Primal,
}

/// CLP-specific solver profile carrying the tunable option surface.
///
/// `ClpProfile` is the associated `Profile` type for [`ClpSolver`](crate::ClpSolver). Other
/// solvers (e.g. `HiGHS`) define their own concrete profile types with their
/// native option names.
///
/// The field defaults are tuned for deterministic, warm-started repeated
/// re-solves: perturbation is off (`102`, not CLP's own `100`=auto default),
/// scaling is off (the cobre prescaler conditions the matrix), and the
/// feasibility tolerances match `HighsProfile` bit-for-bit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClpProfile {
    /// CLP perturbation mode (`Clp_setPerturbation`). `102` disables
    /// perturbation (required for deterministic, reproducible re-solves);
    /// CLP's own default `100` requests automatic perturbation.
    pub perturbation: i32,
    /// CLP scaling mode (`Clp_scaling`). `0` disables scaling because the
    /// cobre prescaler already normalizes matrix entries, mirroring `HiGHS`
    /// `simplex_scale_strategy=0`.
    pub scaling: i32,
    /// Primal feasibility tolerance (`Clp_setPrimalTolerance`).
    pub primal_feasibility_tolerance: f64,
    /// Dual feasibility tolerance (`Clp_setDualTolerance`).
    pub dual_feasibility_tolerance: f64,
    /// Per-attempt simplex iteration cap (`Clp_setMaximumIterations`).
    /// `DEFAULT_PROFILE_HEURISTIC_SENTINEL` (0) selects the per-call heuristic.
    pub simplex_iteration_limit: u32,
    /// Simplex algorithm `solve` dispatches on: dual (`Clp_dual`) or primal
    /// (`Clp_primal`). Read at solve time only; `apply_profile` issues no solve.
    /// Defaults to [`ClpAlgorithm::Dual`].
    pub algorithm: ClpAlgorithm,
    /// Dual-simplex row-pricing mode (drives `cobre_clp_set_dual_row_steepest`).
    ///
    /// Selects how the dual simplex chooses the leaving variable at each
    /// iteration. `1` pins full dual steepest-edge pricing (exact reference
    /// weights, more arithmetic per iteration but typically fewer iterations);
    /// the default `3` is CLP's own steepest-edge constructor default (a
    /// partial / device-style variant). [`SolverInterface::apply_profile`](crate::SolverInterface::apply_profile) drives
    /// this knob: any value other than the `3` sentinel installs that pricing
    /// rule through the shim, while `3` issues no shim call (keeping the default
    /// profile byte-identical to a build that never set pricing).
    pub dual_pricing_mode: i32,
    /// Refactorization cadence (drives `cobre_clp_set_factorization_frequency`).
    ///
    /// Controls how many simplex iterations elapse between full re-factorizations
    /// of the basis matrix. A larger cadence amortizes the refactorization cost
    /// across more iterations at the risk of accumulated numerical drift in the
    /// incremental factor updates. The sentinel `0` means "leave CLP's internal
    /// default in place — do not override".
    /// [`SolverInterface::apply_profile`](crate::SolverInterface::apply_profile)
    /// drives this knob: any non-zero value sets the cadence through the shim,
    /// while `0` issues no shim call.
    pub factorization_frequency: i32,
}

impl Default for ClpProfile {
    fn default() -> Self {
        Self {
            perturbation: 102,
            scaling: 0,
            primal_feasibility_tolerance: 1e-9,
            dual_feasibility_tolerance: 1e-9,
            simplex_iteration_limit: DEFAULT_PROFILE_HEURISTIC_SENTINEL,
            algorithm: ClpAlgorithm::Dual,
            dual_pricing_mode: 3,
            factorization_frequency: 0,
        }
    }
}
