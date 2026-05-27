//! Visited forward-pass states for dominated cut selection.
//!
//! When [`CutSelectionStrategy::Dominated`](crate::CutSelectionStrategy::Dominated)
//! is active, the training loop archives the trial-point state vectors produced
//! by each forward pass so that the domination test can evaluate every cut at
//! every visited point.
//!
//! The archive is organised as one [`StageStates`] per stage.  Each
//! `StageStates` stores its state vectors in a single flat `Vec<f64>` for
//! cache-friendly iteration during the domination sweep.

/// Single-stage visited-states buffer.
///
/// Stores forward-pass trial points as a flat contiguous `Vec<f64>`.
/// Entry `i * state_dimension .. (i + 1) * state_dimension` holds state `i`.
#[derive(Debug, Clone)]
pub struct StageStates {
    /// Flat buffer of accumulated state vectors.
    data: Vec<f64>,
    /// Number of states currently stored.
    count: usize,
    /// Length of each state vector.
    state_dimension: usize,
}

impl StageStates {
    /// Creates a new single-stage buffer, pre-allocating space for
    /// `capacity_states` state vectors of length `state_dimension`.
    #[must_use]
    pub fn new(state_dimension: usize, capacity_states: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity_states * state_dimension),
            count: 0,
            state_dimension,
        }
    }

    /// Returns the number of states currently stored.
    #[must_use]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Returns the dimension of each state vector.
    #[must_use]
    pub fn state_dimension(&self) -> usize {
        self.state_dimension
    }

    /// Append `total_fwd` state vectors from `gathered` into this stage's
    /// buffer.
    ///
    /// `gathered` is a flat slice of length `total_fwd * state_dimension`,
    /// produced by `ExchangeBuffers::gathered_states()`.
    ///
    /// # Panics (debug only)
    ///
    /// Panics if `gathered.len() != total_fwd * self.state_dimension`.
    pub fn append(&mut self, gathered: &[f64], total_fwd: usize) {
        debug_assert_eq!(gathered.len(), total_fwd * self.state_dimension);
        self.data.extend_from_slice(gathered);
        self.count += total_fwd;
    }

    /// Return the flat slice of all accumulated states.
    ///
    /// Length is `self.count * self.state_dimension`.
    #[must_use]
    pub fn states(&self) -> &[f64] {
        &self.data[..self.count * self.state_dimension]
    }

    /// Retain only the most recent `window_states` state vectors.
    ///
    /// Drains the oldest `(count - window_states)` state vectors from the
    /// beginning of the buffer using `Vec::drain(..n)`. The drain shifts the
    /// remaining elements left in O(retained) time, which is acceptable
    /// because trimming runs at the cut-selection cadence (once per
    /// `check_frequency` iterations), not per cut or per state.
    ///
    /// If `count <= window_states`, this is a no-op.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // 100 states of dimension 2, keep last 30.
    /// let mut s = StageStates::new(2, 100);
    /// // ... append states ...
    /// s.trim_to_window(30);
    /// assert_eq!(s.count(), 30);
    /// ```
    pub fn trim_to_window(&mut self, window_states: usize) {
        if self.count <= window_states {
            return;
        }
        let to_remove = self.count - window_states;
        let drain_len = to_remove * self.state_dimension;
        debug_assert!(drain_len <= self.data.len());
        self.data.drain(..drain_len);
        self.count = window_states;
        debug_assert_eq!(self.data.len(), self.count * self.state_dimension);
    }
}

/// Multi-stage archive of visited forward-pass states.
///
/// One [`StageStates`] per stage. Allocated whenever any
/// [`CutSelectionStrategy`] variant is enabled, because the unified
/// value-evaluation kernel evaluates every populated cut at every
/// state in this archive. Also allocated when state export is
/// requested via
/// [`EventConfig::export_states`](crate::config::EventConfig::export_states).
#[derive(Debug, Clone)]
pub struct VisitedStatesArchive {
    stages: Vec<StageStates>,
    /// Number of forward-pass states added per iteration (gathered across all
    /// MPI ranks). Used by [`Self::trim_to_window`] to convert an iteration
    /// window into a state count.
    total_forward_passes: usize,
}

impl VisitedStatesArchive {
    /// Maximum number of state vectors to pre-allocate per stage.
    ///
    /// Prevents excessive upfront virtual memory reservation when
    /// `max_iterations * total_forward_passes` is very large (e.g., 1000 × 100).
    /// The `Vec` will grow beyond this cap on demand via its doubling strategy.
    const MAX_INITIAL_CAPACITY: usize = 4096;

    /// Creates a new archive with one [`StageStates`] per stage.
    ///
    /// Each stage buffer is pre-allocated for up to
    /// `max_iterations * total_forward_passes` state vectors, capped at
    /// `MAX_INITIAL_CAPACITY` to avoid excessive virtual memory
    /// reservation on large configurations. The underlying `Vec` will grow
    /// beyond the cap if needed.
    #[must_use]
    pub fn new(
        num_stages: usize,
        state_dimension: usize,
        max_iterations: u64,
        total_forward_passes: usize,
    ) -> Self {
        let total_states = usize::try_from(max_iterations)
            .unwrap_or(usize::MAX)
            .saturating_mul(total_forward_passes);
        let capacity_per_stage = total_states.min(Self::MAX_INITIAL_CAPACITY);
        let stages = (0..num_stages)
            .map(|_| StageStates::new(state_dimension, capacity_per_stage))
            .collect();
        Self {
            stages,
            total_forward_passes,
        }
    }

    /// Returns the number of stages in the archive.
    #[must_use]
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Returns a shared reference to the [`StageStates`] for `stage`.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= self.num_stages()`.
    #[must_use]
    pub fn stage(&self, stage: usize) -> &StageStates {
        &self.stages[stage]
    }

    /// Returns a mutable reference to the [`StageStates`] for `stage`.
    ///
    /// # Panics
    ///
    /// Panics if `stage >= self.num_stages()`.
    pub fn stage_mut(&mut self, stage: usize) -> &mut StageStates {
        &mut self.stages[stage]
    }

    /// Archive one iteration's gathered states for a specific stage.
    ///
    /// Called in the backward pass after `exchange.exchange()` produces
    /// the gathered buffer for stage `t`.
    pub fn archive_gathered_states(&mut self, stage: usize, gathered: &[f64], total_fwd: usize) {
        self.stages[stage].append(gathered, total_fwd);
    }

    /// Return the flat state slice for a stage.
    ///
    /// Used by `select_for_stage` during cut selection.
    #[must_use]
    pub fn states_for_stage(&self, stage: usize) -> &[f64] {
        self.stages[stage].states()
    }

    /// Number of states accumulated at a given stage.
    #[must_use]
    pub fn count(&self, stage: usize) -> usize {
        self.stages[stage].count()
    }

    /// Trim each stage's buffer so it retains only the most recent
    /// `window_iterations` iterations' worth of forward-pass states.
    ///
    /// Internally this converts the iteration window into a state count by
    /// multiplying by `total_forward_passes` (the gathered forward passes per
    /// iteration across all MPI ranks, captured at construction time), then
    /// delegates to [`StageStates::trim_to_window`] for each stage.
    ///
    /// Trimming is a no-op for stages whose `count <= window_iterations *
    /// total_forward_passes`.
    ///
    /// The training loop is expected to call this at the same cadence as the
    /// cut-selection check (i.e. every `check_frequency` iterations), so the
    /// archive size stays bounded by `window_iterations * total_forward_passes`
    /// regardless of total training length.
    pub fn trim_to_window(&mut self, window_iterations: u64) {
        let window_states = usize::try_from(window_iterations)
            .unwrap_or(usize::MAX)
            .saturating_mul(self.total_forward_passes);
        for stage in &mut self.stages {
            stage.trim_to_window(window_states);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{StageStates, VisitedStatesArchive};

    /// Build a synthetic gathered buffer: `base, base+1, ..., base + total_fwd*state_dim - 1`.
    #[allow(clippy::cast_precision_loss)]
    fn make_gathered(state_dim: usize, total_fwd: usize, base: f64) -> Vec<f64> {
        (0..total_fwd * state_dim)
            .map(|i| base + i as f64)
            .collect()
    }

    #[test]
    fn stage_states_new_preallocates() {
        let s = StageStates::new(4, 100);
        assert!(s.states().is_empty());
        assert_eq!(s.count(), 0);
        assert_eq!(s.state_dimension(), 4);
        // Vec capacity is at least what we asked for.
        assert!(s.data.capacity() >= 400);
    }

    #[test]
    fn stage_states_append_single_batch() {
        let mut s = StageStates::new(2, 10);
        let gathered = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        s.append(&gathered, 3);
        assert_eq!(s.count(), 3);
        assert_eq!(s.states(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn stage_states_append_multiple_batches() {
        let mut s = StageStates::new(2, 10);
        // Batch 1: 3 states.
        let g1 = make_gathered(2, 3, 0.0);
        s.append(&g1, 3);
        // Batch 2: 2 states.
        let g2 = make_gathered(2, 2, 100.0);
        s.append(&g2, 2);
        assert_eq!(s.count(), 5);
        assert_eq!(
            s.states(),
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 101.0, 102.0, 103.0]
        );
    }

    #[test]
    fn stage_states_empty_states() {
        let s = StageStates::new(3, 50);
        assert_eq!(s.states(), &[] as &[f64]);
        assert_eq!(s.count(), 0);
    }

    #[test]
    fn archive_new_creates_correct_stages() {
        let a = VisitedStatesArchive::new(5, 4, 10, 20);
        assert_eq!(a.num_stages(), 5);
        for t in 0..5 {
            assert_eq!(a.count(t), 0);
            assert!(a.states_for_stage(t).is_empty());
        }
    }

    #[test]
    fn archive_gathered_states_delegates() {
        let mut a = VisitedStatesArchive::new(4, 3, 10, 10);
        let gathered = make_gathered(3, 3, 1.0);
        a.archive_gathered_states(2, &gathered, 3);
        assert_eq!(a.count(2), 3);
        assert_eq!(a.count(0), 0);
        assert_eq!(a.count(1), 0);
        assert_eq!(a.count(3), 0);
    }

    #[test]
    fn archive_accumulates_across_iterations() {
        let mut a = VisitedStatesArchive::new(3, 2, 10, 5);
        let g1 = make_gathered(2, 5, 0.0);
        a.archive_gathered_states(1, &g1, 5);
        let g2 = make_gathered(2, 5, 100.0);
        a.archive_gathered_states(1, &g2, 5);
        assert_eq!(a.count(1), 10);
    }

    #[test]
    fn archive_states_for_stage_returns_flat_slice() {
        let mut a = VisitedStatesArchive::new(3, 2, 10, 10);
        let gathered = vec![10.0, 20.0, 30.0, 40.0];
        a.archive_gathered_states(1, &gathered, 2);
        assert_eq!(a.states_for_stage(1), &[10.0, 20.0, 30.0, 40.0]);
        assert!(a.states_for_stage(0).is_empty());
        assert!(a.states_for_stage(2).is_empty());
    }

    // -- StageStates::trim_to_window ------------------------------------

    // AC1: 100 states of dimension 2 trimmed with window 30 keeps the LAST 30
    // states (elements 140..200 of the original flat buffer).
    #[test]
    fn stage_states_trim_to_window_drops_oldest_states() {
        let state_dim = 2;
        let total = 100;
        let mut s = StageStates::new(state_dim, total);
        let gathered = make_gathered(state_dim, total, 0.0);
        s.append(&gathered, total);
        assert_eq!(s.count(), 100);
        assert_eq!(s.states().len(), 200);

        s.trim_to_window(30);

        assert_eq!(s.count(), 30);
        assert_eq!(s.states().len(), 60);
        // Retained slice equals the tail of the original buffer
        // (elements 140..200, i.e. 140.0..199.0).
        let expected: Vec<f64> = (140..200).map(f64::from).collect();
        assert_eq!(s.states(), expected.as_slice());
    }

    // AC2: trim_to_window with window strictly greater than count is a no-op.
    #[test]
    fn stage_states_trim_to_window_noop_when_count_below_window() {
        let state_dim = 2;
        let total = 20;
        let mut s = StageStates::new(state_dim, total);
        let gathered = make_gathered(state_dim, total, 0.0);
        s.append(&gathered, total);
        assert_eq!(s.count(), 20);
        let before: Vec<f64> = s.states().to_vec();

        s.trim_to_window(50);

        assert_eq!(s.count(), 20);
        assert_eq!(s.states(), before.as_slice());
    }

    #[test]
    fn stage_states_trim_to_window_count_equals_window_is_noop() {
        let state_dim = 3;
        let total = 7;
        let mut s = StageStates::new(state_dim, total);
        let gathered = make_gathered(state_dim, total, 10.0);
        s.append(&gathered, total);
        let before: Vec<f64> = s.states().to_vec();

        s.trim_to_window(7);

        assert_eq!(s.count(), 7);
        assert_eq!(s.states(), before.as_slice());
    }

    #[test]
    fn stage_states_trim_to_window_to_zero_clears_buffer() {
        let state_dim = 2;
        let total = 5;
        let mut s = StageStates::new(state_dim, total);
        let gathered = make_gathered(state_dim, total, 0.0);
        s.append(&gathered, total);

        s.trim_to_window(0);

        assert_eq!(s.count(), 0);
        assert!(s.states().is_empty());
    }

    // Trim followed by append must preserve data integrity: the retained
    // window stays at the head, and the newly appended states sit at the tail.
    #[test]
    fn stage_states_trim_then_append_preserves_data() {
        let state_dim = 2;
        let mut s = StageStates::new(state_dim, 100);
        // Initial: 10 states with base 0.0  (values 0..20).
        s.append(&make_gathered(state_dim, 10, 0.0), 10);
        // Trim down to last 4 states (elements 12..20 of original).
        s.trim_to_window(4);
        assert_eq!(s.count(), 4);
        let retained: Vec<f64> = (12..20).map(f64::from).collect();
        assert_eq!(s.states(), retained.as_slice());

        // Append 3 new states starting at 100.0.
        s.append(&make_gathered(state_dim, 3, 100.0), 3);
        assert_eq!(s.count(), 7);

        // Final layout = retained tail + newly appended states.
        let mut expected = retained;
        expected.extend((0..6).map(|i| 100.0 + f64::from(i)));
        assert_eq!(s.states(), expected.as_slice());
    }

    // -- VisitedStatesArchive::trim_to_window ---------------------------

    // AC3: an archive with 5 stages of 100 states each, trimmed with
    // window=3 and total_fwd=10, leaves each stage with at most 30 states.
    #[test]
    fn archive_trim_to_window_trims_each_stage() {
        let num_stages = 5;
        let state_dim = 2;
        let total_fwd = 10;
        let mut a = VisitedStatesArchive::new(num_stages, state_dim, 10, total_fwd);

        // Fill each stage with 100 states (10 iterations * 10 forward passes).
        for t in 0..num_stages {
            for it in 0..10_i32 {
                let base = f64::from(i32::try_from(t).unwrap()) * 1000.0 + f64::from(it) * 100.0;
                let gathered = make_gathered(state_dim, total_fwd, base);
                a.archive_gathered_states(t, &gathered, total_fwd);
            }
            assert_eq!(a.count(t), 100);
        }

        a.trim_to_window(3);

        for t in 0..num_stages {
            assert!(a.count(t) <= 30, "stage {t} has count {}", a.count(t));
            assert_eq!(a.count(t), 30);
            assert_eq!(a.states_for_stage(t).len(), 30 * state_dim);
        }
    }

    // Archive trim is a no-op when count is already within the window.
    #[test]
    fn archive_trim_to_window_noop_when_within_window() {
        let total_fwd = 10;
        let mut a = VisitedStatesArchive::new(2, 2, 10, total_fwd);
        // 2 iterations * 10 forward passes = 20 states per stage.
        for it in 0..2_i32 {
            let gathered = make_gathered(2, total_fwd, f64::from(it) * 100.0);
            a.archive_gathered_states(0, &gathered, total_fwd);
            a.archive_gathered_states(1, &gathered, total_fwd);
        }
        let before_0: Vec<f64> = a.states_for_stage(0).to_vec();
        let before_1: Vec<f64> = a.states_for_stage(1).to_vec();

        // window=5 -> window_states = 50 > 20.
        a.trim_to_window(5);

        assert_eq!(a.count(0), 20);
        assert_eq!(a.count(1), 20);
        assert_eq!(a.states_for_stage(0), before_0.as_slice());
        assert_eq!(a.states_for_stage(1), before_1.as_slice());
    }

    // Archive trim retains the most recent states (verifies tail semantics,
    // not just count).
    #[test]
    fn archive_trim_to_window_retains_most_recent() {
        let state_dim = 2;
        let total_fwd = 5;
        let mut a = VisitedStatesArchive::new(1, state_dim, 10, total_fwd);

        // 4 iterations, each adding 5 states with a distinct base.
        // iter 0: base 0.0   -> values 0..10
        // iter 1: base 100.0 -> values 100..110
        // iter 2: base 200.0 -> values 200..210
        // iter 3: base 300.0 -> values 300..310
        for it in 0..4_i32 {
            let base = f64::from(it) * 100.0;
            a.archive_gathered_states(0, &make_gathered(state_dim, total_fwd, base), total_fwd);
        }
        assert_eq!(a.count(0), 20);

        // Keep last 2 iterations (10 states).
        a.trim_to_window(2);
        assert_eq!(a.count(0), 10);

        // Expected: values from iterations 2 and 3.
        let mut expected: Vec<f64> = (200..210).map(f64::from).collect();
        expected.extend((300..310).map(f64::from));
        assert_eq!(a.states_for_stage(0), expected.as_slice());
    }

    // Trim with window=0 clears every stage in the archive.
    #[test]
    fn archive_trim_to_window_zero_clears_all_stages() {
        let total_fwd = 4;
        let mut a = VisitedStatesArchive::new(3, 2, 10, total_fwd);
        for t in 0..3 {
            a.archive_gathered_states(t, &make_gathered(2, total_fwd, 0.0), total_fwd);
        }

        a.trim_to_window(0);

        for t in 0..3 {
            assert_eq!(a.count(t), 0);
            assert!(a.states_for_stage(t).is_empty());
        }
    }
}
