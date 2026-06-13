//! # cobre-solver
//!
//! LP/MIP solver abstraction for the [Cobre](https://github.com/cobre-rs/cobre) power systems ecosystem.
//!
//! This crate defines a backend-agnostic interface for mathematical programming
//! solvers, with a default [HiGHS](https://highs.dev) backend:
//!
//! - **Solver trait**: unified API for LP and MIP problem construction, solving,
//!   and dual/basis extraction.
//! - **`HiGHS` backend** (`highs` feature, on by default): production-grade
//!   open-source solver, well-suited for iterative LP solving in power system
//!   optimization; exposed as `HighsSolver` and `HighsProfile`.
//! - **`CLP` backend** (`clp` feature, off by default): optional CLP/`CoinUtils`
//!   backend exposed as `ClpSolver` and `ClpProfile`; conformance-validated
//!   as a drop-in implementing the same [`SolverInterface`]. Requires the Clp
//!   and `CoinUtils` submodules to be initialized (`git submodule update --init
//!   --recursive`) before the first build with `--features clp`.
//! - **Basis management**: warm-starting support for iterative algorithms
//!   that solve sequences of related LPs.
//!
//! ## Status
//!
//! This crate is in early development. The API **will** change.
//!
//! See the [repository](https://github.com/cobre-rs/cobre) for the current status.

// Relax strict production lints for test builds. These lints (unwrap_used,
// expect_used, etc.) guard library code but are normal in tests.
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::float_cmp,
        clippy::panic,
        clippy::too_many_lines
    )
)]

// Exactly one LP backend must be selected at compile time. Enabling both
// `highs` and `clp` is rejected with an actionable diagnostic naming the exact
// CLP build command; selecting neither is rejected to avoid downstream
// missing-symbol errors.
#[cfg(all(feature = "highs", feature = "clp"))]
compile_error!(
    "enable exactly one LP backend: `highs` OR `clp`. \
     To use CLP, build with `--no-default-features --features clp`."
);

#[cfg(not(any(feature = "highs", feature = "clp")))]
compile_error!("no LP backend selected: enable exactly one of `highs` or `clp`.");

pub mod ffi;

#[cfg(feature = "clp")]
pub use ffi::clp as clp_ffi;

pub mod trait_def;
pub use trait_def::SolverInterface;

pub mod types;
pub use types::{
    Basis, LpSolution, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate,
};

pub mod profile;
pub use profile::{DEFAULT_PROFILE_HEURISTIC_SENTINEL, DEFAULT_PROFILE_IPM_UNBOUNDED_SENTINEL};

pub mod baking;
pub use baking::{BakingScratch, bake_rows_into_template};

pub mod backends;
pub use backends::profiled::ProfiledSolver;

#[cfg(feature = "highs")]
pub use backends::highs::{HighsProfile, HighsSolver, highs_version};

#[cfg(feature = "clp")]
pub use backends::clp::{ClpAlgorithm, ClpProfile, ClpSolver, clp_version};

// Module-path shims preserving the pre-relocation public paths
// `cobre_solver::highs` / `cobre_solver::clp` for downstream code that imports a
// backend by module rather than through the curated re-exports above. Mirrors
// the `clp_ffi` shim for the `ffi::clp` module.
#[cfg(feature = "clp")]
pub use backends::clp;
#[cfg(feature = "highs")]
pub use backends::highs;

// Active backend selection (compile-time type alias).

/// The compile-time-selected active LP solver backend.
///
/// Downstream crates reference this alias instead of a concrete backend type,
/// so a single binary is bound to exactly one solver with zero runtime
/// dispatch. Exactly one LP backend is enabled at a time — enabling both
/// `highs` and `clp` is rejected at compile time (see the `compile_error!` in
/// this module). This resolves to `ClpSolver` under `--features clp` and to
/// `HighsSolver` under the default `highs` feature. When neither feature is
/// enabled, this alias is not defined.
#[cfg(feature = "clp")]
pub type ActiveSolver = ClpSolver;

/// The compile-time-selected active LP solver backend.
///
/// Downstream crates reference this alias instead of a concrete backend type,
/// so a single binary is bound to exactly one solver with zero runtime
/// dispatch. Exactly one LP backend is enabled at a time — enabling both
/// `highs` and `clp` is rejected at compile time (see the `compile_error!` in
/// this module). This resolves to `ClpSolver` under `--features clp` and to
/// `HighsSolver` under the default `highs` feature. When neither feature is
/// enabled, this alias is not defined.
#[cfg(all(feature = "highs", not(feature = "clp")))]
pub type ActiveSolver = HighsSolver;

/// The solver profile type of the compile-time-selected active backend.
///
/// Resolved under the same mutually-exclusive backend contract as
/// [`ActiveSolver`] (enabling both backends is a compile error): this is
/// `ClpProfile` under `--features clp`, otherwise `HighsProfile` under the
/// default `highs` feature. When neither feature is enabled, this alias is not
/// defined. It is the same type as `<ActiveSolver as SolverInterface>::Profile`.
#[cfg(feature = "clp")]
pub type ActiveProfile = ClpProfile;

/// The solver profile type of the compile-time-selected active backend.
///
/// Resolved under the same mutually-exclusive backend contract as
/// [`ActiveSolver`] (enabling both backends is a compile error): this is
/// `ClpProfile` under `--features clp`, otherwise `HighsProfile` under the
/// default `highs` feature. When neither feature is enabled, this alias is not
/// defined. It is the same type as `<ActiveSolver as SolverInterface>::Profile`.
#[cfg(all(feature = "highs", not(feature = "clp")))]
pub type ActiveProfile = HighsProfile;

/// Returns the version string of the compile-time-selected active backend.
///
/// A single backend-agnostic entry point for metadata wiring. Resolved under
/// the same mutually-exclusive backend contract as [`ActiveSolver`]: returns
/// `clp_version`'s value when CLP is active, `highs_version`'s value when
/// `HiGHS` is active.
#[cfg(feature = "clp")]
#[must_use]
pub fn active_solver_version() -> String {
    clp_version()
}

/// Returns the version string of the compile-time-selected active backend.
///
/// A single backend-agnostic entry point for metadata wiring. Resolved under
/// the same mutually-exclusive backend contract as [`ActiveSolver`]: returns
/// `clp_version`'s value when CLP is active, `highs_version`'s value when
/// `HiGHS` is active.
#[cfg(all(feature = "highs", not(feature = "clp")))]
#[must_use]
pub fn active_solver_version() -> String {
    highs_version()
}

/// Returns the display name of the compile-time-selected active backend.
///
/// This is the same string the active backend's [`SolverInterface::name`]
/// returns (`"CLP"` or `"HiGHS"`). Resolved under the same mutually-exclusive
/// backend contract as [`ActiveSolver`].
#[cfg(feature = "clp")]
#[must_use]
pub fn active_solver_name() -> &'static str {
    "CLP"
}

/// Returns the display name of the compile-time-selected active backend.
///
/// This is the same string the active backend's [`SolverInterface::name`]
/// returns (`"CLP"` or `"HiGHS"`). Resolved under the same mutually-exclusive
/// backend contract as [`ActiveSolver`].
#[cfg(all(feature = "highs", not(feature = "clp")))]
#[must_use]
pub fn active_solver_name() -> &'static str {
    "HiGHS"
}

/// Returns the canonical lowercase backend id used in persisted output metadata
/// (`OutputContext.solver`).
///
/// Unlike [`active_solver_name`] (the mixed-case display name `"CLP"`/`"HiGHS"`),
/// this is the stable lowercase id recorded in output manifests. Resolved under
/// the same mutually-exclusive backend contract as [`ActiveSolver`].
#[cfg(feature = "clp")]
#[must_use]
pub fn active_solver_metadata_id() -> &'static str {
    "clp"
}

/// Returns the canonical lowercase backend id used in persisted output metadata
/// (`OutputContext.solver`).
///
/// Unlike [`active_solver_name`] (the mixed-case display name `"CLP"`/`"HiGHS"`),
/// this is the stable lowercase id recorded in output manifests. Resolved under
/// the same mutually-exclusive backend contract as [`ActiveSolver`].
#[cfg(all(feature = "highs", not(feature = "clp")))]
#[must_use]
pub fn active_solver_metadata_id() -> &'static str {
    "highs"
}

#[cfg(all(feature = "test-support", feature = "highs"))]
pub mod test_support {
    //! Test-only utilities for configuring solver options from integration tests.
    //!
    //! Do **not** enable this feature in production builds. The re-exported functions
    //! call into the `HiGHS` C API directly and bypass all safe-wrapper validation,
    //! so the module is HiGHS-only (gated on both `test-support` and `highs`).

    pub use crate::ffi::{
        cobre_highs_get_double_option, cobre_highs_get_int_option, cobre_highs_set_double_option,
        cobre_highs_set_int_option, cobre_highs_set_string_option,
    };
}

#[cfg(all(test, any(feature = "highs", feature = "clp")))]
mod active_alias_tests {
    use crate::{
        ActiveProfile, ActiveSolver, SolverInterface, active_solver_metadata_id,
        active_solver_name, active_solver_version,
    };

    /// `ActiveProfile` must be exactly the profile of the active solver.
    #[allow(dead_code)]
    const fn _assert_profile_identity(
        p: ActiveProfile,
    ) -> <ActiveSolver as SolverInterface>::Profile {
        p
    }

    #[test]
    fn active_aliases_resolve_per_feature() {
        let name = active_solver_name();
        let version = active_solver_version();
        assert!(!version.is_empty(), "active version must be non-empty");
        assert!(
            version.contains('.'),
            "active version `{version}` should contain a `.`"
        );

        // Exactly one backend is active (both-enabled is a compile error): the
        // active backend is CLP under `--features clp`, else HiGHS by default.
        #[cfg(feature = "clp")]
        assert_eq!(name, "CLP");
        #[cfg(all(feature = "highs", not(feature = "clp")))]
        assert_eq!(name, "HiGHS");
    }

    #[test]
    fn active_solver_name_matches_instance_name() {
        let solver = ActiveSolver::new().expect("active solver must construct");
        assert_eq!(solver.name(), active_solver_name());
    }

    /// HiGHS-only build (`--features highs`, no `clp`): the active backend
    /// resolves to `HiGHS`. The both-on configuration cannot compile (a
    /// `compile_error!` in `lib.rs` rejects it), so no runnable test is needed
    /// for that contract.
    #[cfg(all(feature = "highs", not(feature = "clp")))]
    #[test]
    fn active_backend_is_highs_when_only_highs_enabled() {
        assert_eq!(active_solver_name(), "HiGHS");
        assert_eq!(active_solver_metadata_id(), "highs");
    }

    /// CLP build (`--no-default-features --features clp`): the active backend
    /// resolves to CLP.
    #[cfg(feature = "clp")]
    #[test]
    fn active_backend_is_clp_when_clp_enabled() {
        assert_eq!(active_solver_name(), "CLP");
        assert_eq!(active_solver_metadata_id(), "clp");
    }
}
