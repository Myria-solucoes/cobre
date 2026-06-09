//! Generic `ProfiledSolver<S>` wrapper with per-phase LP-solver configuration.
//!
//! [`ProfiledSolver`] wraps any [`SolverInterface`] implementor, tracks the
//! currently-applied solver profile (typed as `S::Profile`), and skips FFI
//! option-setter calls when the new profile matches the current one
//! (delta-only dispatch via `PartialEq`). All other [`SolverInterface`] methods
//! are transparently forwarded to the inner solver.
//!
//! # Construction
//!
//! `ProfiledSolver::new(inner)` assumes the inner solver is in a state
//! consistent with `S::Profile::default()` and issues no FFI calls.
//!
//! # Usage
//!
//! ```rust
//! # #[cfg(feature = "highs")] {
//! use cobre_solver::{HighsProfile, ProfiledSolver, SolverInterface, HighsSolver};
//!
//! let inner = HighsSolver::new().expect("HiGHS init");
//! let mut solver = ProfiledSolver::new(inner);
//! solver.set_profile(&HighsProfile::default());
//! assert_eq!(solver.current_profile(), &HighsProfile::default());
//! # }
//! ```

use crate::{
    SolverInterface,
    types::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate},
};

/// Wraps any [`SolverInterface`] implementor with per-phase profile
/// configuration.
///
/// Tracks the currently-applied profile (typed as `S::Profile`) and skips
/// no-op FFI calls when the same profile is reapplied (delta-only dispatch
/// via `PartialEq`). The wrapper itself implements [`SolverInterface`] by
/// transparently forwarding all method calls to the inner solver.
/// [`ProfiledSolver::set_profile`] is the only non-trait-method addition.
///
/// # Generic parameter
///
/// `S` must implement [`SolverInterface`]. The wrapper is resolved at compile
/// time (monomorphization) to preserve zero-cost forwarding on the hot path.
pub struct ProfiledSolver<S: SolverInterface> {
    inner: S,
    current_profile: S::Profile,
}

impl<S: SolverInterface> ProfiledSolver<S> {
    /// Wrap an existing solver with the default profile.
    ///
    /// The wrapper does NOT issue any FFI calls on construction — the inner
    /// solver is assumed to be in a state consistent with
    /// `S::Profile::default()`, which is exactly how it has been
    /// constructed historically.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            current_profile: S::Profile::default(),
        }
    }

    /// Apply a new profile to the inner solver.
    ///
    /// If `profile == current_profile`, this method returns immediately with
    /// zero inner method calls. Otherwise it delegates to
    /// `inner.apply_profile(profile)` which issues all necessary FFI calls.
    ///
    /// After the call returns, `current_profile() == profile`.
    ///
    /// # Call-site contract
    ///
    /// Callers invoke this once per phase boundary. It is NOT intended to be
    /// called inside the hot solve loop.
    pub fn set_profile(&mut self, profile: &S::Profile) {
        if *profile == self.current_profile {
            return;
        }
        self.inner.apply_profile(profile);
        self.current_profile = *profile;
    }

    /// Read-only access to the currently applied profile.
    ///
    /// Returns the profile that was last successfully applied via
    /// [`ProfiledSolver::set_profile`], or `S::Profile::default()` if no
    /// profile has been applied yet.
    pub fn current_profile(&self) -> &S::Profile {
        &self.current_profile
    }

    /// Shared reference to the wrapped inner solver.
    ///
    /// Intended for test code and rare adapter sites that need to inspect
    /// mock-specific fields on the inner solver. Not used on the hot path.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Exclusive reference to the wrapped inner solver.
    ///
    /// Intended for test code and rare adapter sites that need to mutate
    /// mock-specific state on the inner solver. Not used on the hot path.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }
}

// Transparent `SolverInterface` forwarding.
impl<S: SolverInterface> SolverInterface for ProfiledSolver<S> {
    type Profile = S::Profile;

    fn apply_profile(&mut self, profile: &S::Profile) {
        self.inner.apply_profile(profile);
        self.current_profile = *profile;
    }

    fn load_model(&mut self, template: &StageTemplate) {
        self.inner.load_model(template);
    }

    fn add_rows(&mut self, rows: &RowBatch) {
        self.inner.add_rows(rows);
    }

    fn set_row_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        self.inner.set_row_bounds(indices, lower, upper);
    }

    fn set_col_bounds(&mut self, indices: &[usize], lower: &[f64], upper: &[f64]) {
        self.inner.set_col_bounds(indices, lower, upper);
    }

    fn solve(&mut self, basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
        // Direct forward: the profile is installed at phase boundaries via
        // `set_profile`, and the retry-escalation path re-applies the full
        // profile (tolerances + strategies) after `restore_default_settings`,
        // so the inner solver's options already equal `current_profile` on
        // entry to every solve. Verified result-neutral against the D01-D15
        // parity hashes (identical actual hashes with and without the former
        // per-solve re-application).
        self.inner.solve(basis)
    }

    fn get_basis(&mut self, out: &mut Basis) {
        self.inner.get_basis(out);
    }

    fn statistics(&self) -> SolverStatistics {
        self.inner.statistics()
    }

    fn statistics_into(&self, out: &mut SolverStatistics) {
        self.inner.statistics_into(out);
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn solver_name_version(&self) -> String {
        self.inner.solver_name_version()
    }

    fn record_reconstruction_stats(&mut self) {
        self.inner.record_reconstruction_stats();
    }

    fn reset_solver_state(&mut self) {
        self.inner.reset_solver_state();
    }

    fn set_primal_feasibility_tolerance(&mut self, value: f64) {
        self.inner.set_primal_feasibility_tolerance(value);
    }

    fn set_dual_feasibility_tolerance(&mut self, value: f64) {
        self.inner.set_dual_feasibility_tolerance(value);
    }

    fn set_simplex_iteration_limit_profile(&mut self, value: u32) {
        self.inner.set_simplex_iteration_limit_profile(value);
    }

    fn set_ipm_iteration_limit_profile(&mut self, value: u32) {
        self.inner.set_ipm_iteration_limit_profile(value);
    }
}

// This test module uses `HighsProfile` with field values for delta-tracking
// assertions, so it cannot use the backend-agnostic fieldless mock profile. It
// is gated behind `feature = "highs"`; `ProfiledSolver`'s delta-tracking is
// backend-agnostic, so exercising it on the HiGHS job is sufficient coverage.
#[cfg(all(test, feature = "highs"))]
mod tests {
    use std::cell::RefCell;

    use super::ProfiledSolver;
    use crate::{
        HighsProfile, SolverInterface,
        types::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate},
    };

    // ── RecordingMockSolver ───────────────────────────────────────────────────

    /// Recorded invocation of a [`SolverInterface`] method.
    #[derive(Debug, Clone, PartialEq)]
    enum RecordedCall {
        LoadModel,
        AddRows,
        SetRowBounds,
        SetColBounds,
        Solve,
        ApplyProfile(HighsProfile),
    }

    /// A minimal [`SolverInterface`] implementor that records every invocation
    /// into an interior-mutable call log.
    ///
    /// Returned `solve` results are always `Err(SolverError::InternalError)`
    /// because the mock does not represent a real LP solver; callers only
    /// inspect the call log in these unit tests.
    struct RecordingMockSolver {
        calls: RefCell<Vec<RecordedCall>>,
    }

    impl RecordingMockSolver {
        fn new() -> Self {
            Self {
                calls: RefCell::new(Vec::new()),
            }
        }

        /// Returns a snapshot of all recorded calls.
        pub(crate) fn recorded_calls(&self) -> Vec<RecordedCall> {
            self.calls.borrow().clone()
        }
    }

    // SAFETY for `Send`: `RecordingMockSolver` is only ever constructed and
    // used on a single thread within these unit tests. `RefCell` is not `Sync`,
    // but the `Send` bound on `SolverInterface` merely permits transferring
    // ownership to another thread — it does not permit concurrent access. The
    // mock is never actually transferred across threads; the `unsafe impl Send`
    // is required by the trait bound and is safe in this single-threaded test
    // context.
    unsafe impl Send for RecordingMockSolver {}

    impl SolverInterface for RecordingMockSolver {
        type Profile = HighsProfile;

        fn apply_profile(&mut self, profile: &HighsProfile) {
            self.calls
                .borrow_mut()
                .push(RecordedCall::ApplyProfile(*profile));
        }

        fn load_model(&mut self, _template: &StageTemplate) {
            self.calls.borrow_mut().push(RecordedCall::LoadModel);
        }

        fn add_rows(&mut self, _rows: &RowBatch) {
            self.calls.borrow_mut().push(RecordedCall::AddRows);
        }

        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {
            self.calls.borrow_mut().push(RecordedCall::SetRowBounds);
        }

        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {
            self.calls.borrow_mut().push(RecordedCall::SetColBounds);
        }

        fn solve(&mut self, _basis: Option<&Basis>) -> Result<SolutionView<'_>, SolverError> {
            self.calls.borrow_mut().push(RecordedCall::Solve);
            Err(SolverError::InternalError {
                message: "mock".to_string(),
                error_code: None,
            })
        }

        fn get_basis(&mut self, _out: &mut Basis) {}

        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }

        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&SolverStatistics::default());
        }

        fn name(&self) -> &'static str {
            "RecordingMock"
        }

        fn solver_name_version(&self) -> String {
            "RecordingMockSolver 0.0.0".to_string()
        }

        fn set_primal_feasibility_tolerance(&mut self, _value: f64) {}

        fn set_dual_feasibility_tolerance(&mut self, _value: f64) {}

        fn set_simplex_iteration_limit_profile(&mut self, _value: u32) {}

        fn set_ipm_iteration_limit_profile(&mut self, _value: u32) {}
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Filter recorded calls to extract only `apply_profile` calls.
    fn filter_profile_calls(calls: &[RecordedCall]) -> Vec<&RecordedCall> {
        calls
            .iter()
            .filter(|c| matches!(c, RecordedCall::ApplyProfile(_)))
            .collect()
    }

    fn make_test_template() -> StageTemplate {
        StageTemplate {
            num_cols: 1,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32, 0],
            row_indices: vec![],
            values: vec![],
            col_lower: vec![0.0],
            col_upper: vec![1.0],
            objective: vec![0.0],
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

    fn make_test_row_batch() -> RowBatch {
        RowBatch {
            num_rows: 0,
            row_starts: vec![0_i32],
            col_indices: vec![],
            values: vec![],
            row_lower: vec![],
            row_upper: vec![],
        }
    }

    /// AC-3: `ProfiledSolver::new` must not dispatch any FFI setter calls.
    #[test]
    fn new_issues_no_ffi_calls() {
        let mock = RecordingMockSolver::new();
        let solver = ProfiledSolver::new(mock);
        let calls = solver.inner.recorded_calls();
        assert!(
            calls.is_empty(),
            "expected zero calls after ProfiledSolver::new, got: {calls:?}"
        );
    }

    /// AC-4: `set_profile` with a profile equal to `current_profile` issues
    /// zero `apply_profile` calls (noop delta-tracking).
    #[test]
    fn set_profile_noop_when_unchanged() {
        let mock = RecordingMockSolver::new();
        let mut solver = ProfiledSolver::new(mock);

        // Apply the default profile — same as the initial `current_profile`.
        solver.set_profile(&HighsProfile::default());

        let calls = solver.inner.recorded_calls();
        let profile_calls = filter_profile_calls(&calls);
        assert!(
            profile_calls.is_empty(),
            "expected zero apply_profile calls when profile unchanged, got: {profile_calls:?}"
        );
    }

    /// AC-5: `set_profile` with any field differing from `current_profile`
    /// dispatches exactly one `apply_profile` call carrying the complete new
    /// profile value.
    ///
    /// In the associated-type design, `apply_profile` is a single atomic call
    /// on the inner solver — there is no per-field dispatch. The noop guard is
    /// purely a `PartialEq` comparison on the whole profile.
    #[test]
    fn set_profile_dispatches_apply_profile_when_changed() {
        // ── sub-test 1: only primal tolerance changed ──
        {
            let mock = RecordingMockSolver::new();
            let mut solver = ProfiledSolver::new(mock);
            let p = HighsProfile {
                primal_feasibility_tolerance: 1e-7,
                ..HighsProfile::default()
            };
            solver.set_profile(&p);
            let calls = solver.inner.recorded_calls();
            let profile_calls = filter_profile_calls(&calls);
            assert_eq!(
                profile_calls,
                vec![&RecordedCall::ApplyProfile(p)],
                "expected one ApplyProfile(p) for primal-only change"
            );
        }

        // ── sub-test 2: only dual tolerance changed ──
        {
            let mock = RecordingMockSolver::new();
            let mut solver = ProfiledSolver::new(mock);
            let p = HighsProfile {
                dual_feasibility_tolerance: 1e-7,
                ..HighsProfile::default()
            };
            solver.set_profile(&p);
            let calls = solver.inner.recorded_calls();
            let profile_calls = filter_profile_calls(&calls);
            assert_eq!(
                profile_calls,
                vec![&RecordedCall::ApplyProfile(p)],
                "expected one ApplyProfile(p) for dual-only change"
            );
        }

        // ── sub-test 3: only simplex cap changed ──
        {
            let mock = RecordingMockSolver::new();
            let mut solver = ProfiledSolver::new(mock);
            let p = HighsProfile {
                simplex_iteration_limit: 50_000,
                ..HighsProfile::default()
            };
            solver.set_profile(&p);
            let calls = solver.inner.recorded_calls();
            let profile_calls = filter_profile_calls(&calls);
            assert_eq!(
                profile_calls,
                vec![&RecordedCall::ApplyProfile(p)],
                "expected one ApplyProfile(p) for simplex-cap-only change"
            );
        }

        // ── sub-test 4: only IPM cap changed ──
        {
            let mock = RecordingMockSolver::new();
            let mut solver = ProfiledSolver::new(mock);
            let p = HighsProfile {
                ipm_iteration_limit: 5_000,
                ..HighsProfile::default()
            };
            solver.set_profile(&p);
            let calls = solver.inner.recorded_calls();
            let profile_calls = filter_profile_calls(&calls);
            assert_eq!(
                profile_calls,
                vec![&RecordedCall::ApplyProfile(p)],
                "expected one ApplyProfile(p) for ipm-cap-only change"
            );
        }

        // ── sub-test 5: only dual edge weight strategy changed ──
        {
            let mock = RecordingMockSolver::new();
            let mut solver = ProfiledSolver::new(mock);
            let p = HighsProfile {
                simplex_dual_edge_weight_strategy: 0, // Dantzig
                ..HighsProfile::default()
            };
            solver.set_profile(&p);
            let calls = solver.inner.recorded_calls();
            let profile_calls = filter_profile_calls(&calls);
            assert_eq!(
                profile_calls,
                vec![&RecordedCall::ApplyProfile(p)],
                "expected one ApplyProfile(p) for dual-edge-weight-only change"
            );
        }

        // ── sub-test 6: only price strategy changed ──
        {
            let mock = RecordingMockSolver::new();
            let mut solver = ProfiledSolver::new(mock);
            let p = HighsProfile {
                simplex_price_strategy: 2, // RowHyperSparse
                ..HighsProfile::default()
            };
            solver.set_profile(&p);
            let calls = solver.inner.recorded_calls();
            let profile_calls = filter_profile_calls(&calls);
            assert_eq!(
                profile_calls,
                vec![&RecordedCall::ApplyProfile(p)],
                "expected one ApplyProfile(p) for price-strategy-only change"
            );
        }
    }

    /// AC-6: When all profile fields differ from the default, `set_profile`
    /// dispatches exactly one `apply_profile` call carrying the complete new
    /// profile.
    #[test]
    fn set_profile_full_change_dispatches_single_apply_profile() {
        let mock = RecordingMockSolver::new();
        let mut solver = ProfiledSolver::new(mock);

        let p = HighsProfile {
            primal_feasibility_tolerance: 1e-7,
            dual_feasibility_tolerance: 1e-7,
            simplex_iteration_limit: 50_000,
            ipm_iteration_limit: 5_000,
            simplex_dual_edge_weight_strategy: 0, // Dantzig
            simplex_scale_strategy: 2,            // Curtis-Reid
            simplex_price_strategy: 2,            // RowHyperSparse
        };
        solver.set_profile(&p);

        let calls = solver.inner.recorded_calls();
        let profile_calls: Vec<_> = filter_profile_calls(&calls).into_iter().cloned().collect();

        assert_eq!(
            profile_calls,
            vec![RecordedCall::ApplyProfile(p)],
            "expected exactly one ApplyProfile call with the complete profile"
        );
    }

    /// AC-7: `ProfiledSolver<S>` forwards `load_model`, `add_rows`,
    /// `set_row_bounds`, `set_col_bounds`, and `solve` transparently to the
    /// inner solver. `solve` delegates directly without re-applying the
    /// profile — the profile is installed at phase boundaries via
    /// `set_profile`, not per solve.
    #[test]
    fn solver_interface_methods_forward_to_inner() {
        let mock = RecordingMockSolver::new();
        let mut solver = ProfiledSolver::new(mock);

        let template = make_test_template();
        let rows = make_test_row_batch();

        solver.load_model(&template);
        solver.add_rows(&rows);
        solver.set_row_bounds(&[], &[], &[]);
        solver.set_col_bounds(&[], &[], &[]);
        let _ = solver.solve(None);

        let calls = solver.inner.recorded_calls();
        assert!(
            calls.contains(&RecordedCall::LoadModel),
            "expected LoadModel in call log, got: {calls:?}"
        );
        assert!(
            calls.contains(&RecordedCall::AddRows),
            "expected AddRows in call log, got: {calls:?}"
        );
        assert!(
            calls.contains(&RecordedCall::SetRowBounds),
            "expected SetRowBounds in call log, got: {calls:?}"
        );
        assert!(
            calls.contains(&RecordedCall::SetColBounds),
            "expected SetColBounds in call log, got: {calls:?}"
        );
        assert!(
            calls.contains(&RecordedCall::Solve),
            "expected Solve in call log, got: {calls:?}"
        );
        // `solve()` forwards directly to the inner solver and does NOT
        // re-apply the profile (that happens at phase boundaries via
        // `set_profile`), so no ApplyProfile call originates from solve().
        let profile_calls = filter_profile_calls(&calls);
        assert!(
            profile_calls.is_empty(),
            "solve() must not trigger an ApplyProfile call, got: {calls:?}"
        );
    }
}
