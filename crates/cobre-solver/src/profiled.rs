//! Generic `ProfiledSolver<S>` wrapper with per-phase LP-solver configuration.
//!
//! [`ProfiledSolver`] wraps any [`SolverInterface`] implementor, tracks the
//! currently-applied [`SolveProfile`], and skips FFI option-setter calls when
//! the new profile matches the current one (delta-only dispatch). All other
//! [`SolverInterface`] methods are transparently forwarded to the inner solver.
//!
//! # Construction
//!
//! `ProfiledSolver::new(inner)` assumes the inner solver is in a state
//! consistent with `SolveProfile::default()` and issues no FFI calls.
//!
//! # Usage
//!
//! ```rust
//! use cobre_solver::{ProfiledSolver, SolveProfile, SolverInterface, HighsSolver};
//!
//! let inner = HighsSolver::new().expect("HiGHS init");
//! let mut solver = ProfiledSolver::new(inner);
//! solver.set_profile(&SolveProfile::default());
//! assert_eq!(solver.current_profile(), &SolveProfile::default());
//! ```

use crate::{
    SolveProfile, SolverInterface,
    types::{Basis, RowBatch, SolutionView, SolverError, SolverStatistics, StageTemplate},
};

/// Wraps any [`SolverInterface`] implementor with per-phase profile
/// configuration.
///
/// Tracks the currently-applied profile and skips no-op FFI calls when the
/// same profile is reapplied. The wrapper itself implements [`SolverInterface`]
/// by transparently forwarding all method calls to the inner solver.
/// [`ProfiledSolver::set_profile`] is the only non-trait-method addition.
///
/// # Generic parameter
///
/// `S` must implement [`SolverInterface`]. The wrapper is resolved at compile
/// time (monomorphization) to preserve zero-cost forwarding on the hot path.
pub struct ProfiledSolver<S: SolverInterface> {
    inner: S,
    current_profile: SolveProfile,
}

impl<S: SolverInterface> ProfiledSolver<S> {
    /// Wrap an existing solver with the default profile.
    ///
    /// The wrapper does NOT issue any FFI calls on construction — the inner
    /// solver is assumed to be in a state consistent with
    /// `SolveProfile::default()`, which is exactly how it has been
    /// constructed historically.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            current_profile: SolveProfile::default(),
        }
    }

    /// Apply a new profile to the inner solver.
    ///
    /// Only fields that differ from `current_profile` trigger trait-method
    /// calls on the inner solver. The dispatch order is deterministic:
    /// primal feasibility → dual feasibility → simplex cap → IPM cap.
    ///
    /// If `profile == current_profile`, this method returns immediately with
    /// zero inner method calls.
    ///
    /// After the call returns, `current_profile() == profile`.
    ///
    /// # Call-site contract
    ///
    /// Callers invoke this once per phase boundary. It is NOT intended to be
    /// called inside the hot solve loop.
    pub fn set_profile(&mut self, profile: &SolveProfile) {
        if *profile == self.current_profile {
            return;
        }
        // Exact bit-equality is intentional: `SolveProfile` is `Copy + PartialEq`
        // precisely so that field-level delta tracking can compare stored vs. requested
        // values without a margin-of-error. The goal is to avoid FFI calls only when
        // the exact same bitpattern was previously applied — not to express a numerical
        // tolerance concept. `float_cmp` is suppressed for this reason.
        #[allow(clippy::float_cmp)]
        if profile.primal_feasibility_tolerance != self.current_profile.primal_feasibility_tolerance
        {
            self.inner
                .set_primal_feasibility_tolerance(profile.primal_feasibility_tolerance);
        }
        #[allow(clippy::float_cmp)]
        if profile.dual_feasibility_tolerance != self.current_profile.dual_feasibility_tolerance {
            self.inner
                .set_dual_feasibility_tolerance(profile.dual_feasibility_tolerance);
        }
        if profile.simplex_iteration_limit != self.current_profile.simplex_iteration_limit {
            self.inner
                .set_simplex_iteration_limit_profile(profile.simplex_iteration_limit);
        }
        if profile.ipm_iteration_limit != self.current_profile.ipm_iteration_limit {
            self.inner
                .set_ipm_iteration_limit_profile(profile.ipm_iteration_limit);
        }
        self.current_profile = *profile;
    }

    /// Read-only access to the currently applied profile.
    ///
    /// Returns the profile that was last successfully applied via
    /// [`ProfiledSolver::set_profile`], or `SolveProfile::default()` if no
    /// profile has been applied yet.
    pub fn current_profile(&self) -> &SolveProfile {
        &self.current_profile
    }

    /// Re-apply the current profile to the inner solver unconditionally.
    ///
    /// Unlike [`set_profile`], which skips FFI calls when the profile is
    /// unchanged, this method always dispatches all four setter calls. It is
    /// used internally before each solve to survive any `HiGHS` internal
    /// option reset that may occur between solves.
    fn apply_profile(&mut self) {
        self.inner
            .set_primal_feasibility_tolerance(self.current_profile.primal_feasibility_tolerance);
        self.inner
            .set_dual_feasibility_tolerance(self.current_profile.dual_feasibility_tolerance);
        self.inner
            .set_simplex_iteration_limit_profile(self.current_profile.simplex_iteration_limit);
        self.inner
            .set_ipm_iteration_limit_profile(self.current_profile.ipm_iteration_limit);
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
        // Re-apply profile before solve to survive any HiGHS internal option reset.
        self.apply_profile();
        self.inner.solve(basis)
    }

    fn get_basis(&mut self, out: &mut Basis) {
        self.inner.get_basis(out);
    }

    fn statistics(&self) -> SolverStatistics {
        self.inner.statistics()
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

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::ProfiledSolver;
    use crate::{
        SolveProfile, SolverInterface,
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
        SetPrimalFeas(f64),
        SetDualFeas(f64),
        SetSimplexCap(u32),
        SetIpmCap(u32),
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

        fn name(&self) -> &'static str {
            "RecordingMock"
        }

        fn solver_name_version(&self) -> String {
            "RecordingMockSolver 0.0.0".to_string()
        }

        fn set_primal_feasibility_tolerance(&mut self, value: f64) {
            self.calls
                .borrow_mut()
                .push(RecordedCall::SetPrimalFeas(value));
        }

        fn set_dual_feasibility_tolerance(&mut self, value: f64) {
            self.calls
                .borrow_mut()
                .push(RecordedCall::SetDualFeas(value));
        }

        fn set_simplex_iteration_limit_profile(&mut self, value: u32) {
            self.calls
                .borrow_mut()
                .push(RecordedCall::SetSimplexCap(value));
        }

        fn set_ipm_iteration_limit_profile(&mut self, value: u32) {
            self.calls.borrow_mut().push(RecordedCall::SetIpmCap(value));
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Filter recorded calls to extract only profile setter calls.
    fn filter_profile_calls(calls: &[RecordedCall]) -> Vec<&RecordedCall> {
        calls
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    RecordedCall::SetPrimalFeas(_)
                        | RecordedCall::SetDualFeas(_)
                        | RecordedCall::SetSimplexCap(_)
                        | RecordedCall::SetIpmCap(_)
                )
            })
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

    // ── AC-3 ─────────────────────────────────────────────────────────────────

    /// AC-3: `ProfiledSolver::new` must not dispatch any FFI setter calls.
    #[test]
    fn new_issues_no_ffi_calls() {
        let mock = RecordingMockSolver::new();
        let solver = ProfiledSolver::new(mock);
        // Access the inner mock's call log via the wrapper's inner field —
        // since inner is private, we use current_profile() as the construction
        // witness and retrieve the mock after consuming the wrapper.
        let mock = solver.inner;
        let calls = mock.recorded_calls();
        assert!(
            calls.is_empty(),
            "expected zero calls after ProfiledSolver::new, got: {calls:?}"
        );
    }

    // ── AC-4 ─────────────────────────────────────────────────────────────────

    /// AC-4: `set_profile` with a profile equal to `current_profile` issues
    /// zero FFI setter calls.
    #[test]
    fn set_profile_noop_when_unchanged() {
        let mock = RecordingMockSolver::new();
        let mut solver = ProfiledSolver::new(mock);

        // Apply the default profile — same as the initial `current_profile`.
        solver.set_profile(&SolveProfile::default());

        let calls = solver.inner.recorded_calls();
        let profile_calls = filter_profile_calls(&calls);
        assert!(
            profile_calls.is_empty(),
            "expected zero profile setter calls when profile unchanged, got: {profile_calls:?}"
        );
    }

    // ── AC-5 ─────────────────────────────────────────────────────────────────

    /// AC-5: `set_profile` with exactly one field changed dispatches exactly
    /// one setter call matching that field, and no other setter calls.
    #[test]
    fn set_profile_dispatches_only_changed_field() {
        let default = SolveProfile::default();

        // ── sub-test 1: only primal tolerance changed ──
        {
            let mock = RecordingMockSolver::new();
            let mut solver = ProfiledSolver::new(mock);
            let p = SolveProfile {
                primal_feasibility_tolerance: 1e-7,
                ..default
            };
            solver.set_profile(&p);
            let calls = solver.inner.recorded_calls();
            let setter_calls = filter_profile_calls(&calls);
            assert_eq!(
                setter_calls,
                vec![&RecordedCall::SetPrimalFeas(1e-7)],
                "expected only SetPrimalFeas(1e-7) for primal-only change"
            );
        }

        // ── sub-test 2: only dual tolerance changed ──
        {
            let mock = RecordingMockSolver::new();
            let mut solver = ProfiledSolver::new(mock);
            let p = SolveProfile {
                dual_feasibility_tolerance: 1e-7,
                ..default
            };
            solver.set_profile(&p);
            let calls = solver.inner.recorded_calls();
            let setter_calls = filter_profile_calls(&calls);
            assert_eq!(
                setter_calls,
                vec![&RecordedCall::SetDualFeas(1e-7)],
                "expected only SetDualFeas(1e-7) for dual-only change"
            );
        }

        // ── sub-test 3: only simplex cap changed ──
        {
            let mock = RecordingMockSolver::new();
            let mut solver = ProfiledSolver::new(mock);
            let p = SolveProfile {
                simplex_iteration_limit: 50_000,
                ..default
            };
            solver.set_profile(&p);
            let calls = solver.inner.recorded_calls();
            let setter_calls = filter_profile_calls(&calls);
            assert_eq!(
                setter_calls,
                vec![&RecordedCall::SetSimplexCap(50_000)],
                "expected only SetSimplexCap(50_000) for simplex-only change"
            );
        }

        // ── sub-test 4: only IPM cap changed ──
        {
            let mock = RecordingMockSolver::new();
            let mut solver = ProfiledSolver::new(mock);
            let p = SolveProfile {
                ipm_iteration_limit: 5_000,
                ..default
            };
            solver.set_profile(&p);
            let calls = solver.inner.recorded_calls();
            let setter_calls = filter_profile_calls(&calls);
            assert_eq!(
                setter_calls,
                vec![&RecordedCall::SetIpmCap(5_000)],
                "expected only SetIpmCap(5_000) for ipm-only change"
            );
        }
    }

    // ── AC-6 ─────────────────────────────────────────────────────────────────

    /// AC-6: When all four profile fields differ, `set_profile` dispatches
    /// exactly four setter calls in the deterministic order:
    /// `SetPrimalFeas` → `SetDualFeas` → `SetSimplexCap` → `SetIpmCap`.
    #[test]
    fn set_profile_full_change_uses_deterministic_order() {
        let mock = RecordingMockSolver::new();
        let mut solver = ProfiledSolver::new(mock);

        let p = SolveProfile {
            primal_feasibility_tolerance: 1e-7,
            dual_feasibility_tolerance: 1e-7,
            simplex_iteration_limit: 50_000,
            ipm_iteration_limit: 5_000,
        };
        solver.set_profile(&p);

        let calls = solver.inner.recorded_calls();
        let setter_calls: Vec<_> = filter_profile_calls(&calls).into_iter().cloned().collect();

        assert_eq!(
            setter_calls,
            vec![
                RecordedCall::SetPrimalFeas(1e-7),
                RecordedCall::SetDualFeas(1e-7),
                RecordedCall::SetSimplexCap(50_000),
                RecordedCall::SetIpmCap(5_000),
            ],
            "setter calls must appear in deterministic order: primal, dual, simplex, ipm"
        );
    }

    // ── AC-7 ─────────────────────────────────────────────────────────────────

    /// AC-7: `ProfiledSolver<S>` forwards `load_model`, `add_rows`,
    /// `set_row_bounds`, `set_col_bounds`, and `solve` transparently to the
    /// inner solver.
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
    }
}
