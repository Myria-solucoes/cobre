//! Future Cost Function (FCF) — all-stages container for Benders cuts.
//!
//! [`FutureCostFunction`] wraps one [`CutPool`] per stage as a thin orchestration
//! layer over the training loop's cut operations.
//!
//! ## Stage indexing
//!
//! FCF methods use **0-based** stage indices; the SDDP spec uses 1-based stage
//! numbers (`t = 1..T`). Callers convert with `stage = t - 1`.
//!
//! ## Memory allocation
//!
//! All pool storage is pre-allocated by [`FutureCostFunction::new`] — no heap
//! allocation on the training-loop hot path.
//!
//! ## Example
//!
//! ```rust
//! use cobre_sddp::cut::fcf::FutureCostFunction;
//!
//! let mut fcf = FutureCostFunction::new(3, 4, 10, 50, &[0; 3]);
//! assert_eq!(fcf.pools.len(), 3);
//!
//! let coeffs = vec![1.0, 0.0, 0.0, 0.0];
//! fcf.add_cut(1, 0, 0, 5.0, &coeffs);
//!
//! let cuts: Vec<_> = fcf.active_cuts(1).collect();
//! assert_eq!(cuts.len(), 1);
//! assert_eq!(fcf.total_active_cuts(), 1);
//! assert_eq!(fcf.cuts_in_lp(1), 1);
//! ```

use super::pool::CutPool;
use crate::FullFcf;
use crate::PolicyLoadProof;
use crate::SddpError;
use crate::SddpError::Validation;

use cobre_io::StageCutsReadResult;

/// All-stages container for the Future Cost Function (FCF): one [`CutPool`] per
/// stage. Per-cut logic is delegated to [`CutPool`].
///
/// Each pool carries its own [`CutPool::state_dimension`]; a stage whose cuts
/// span fewer state dimensions (a successor with `inflow_lags: false`) sizes a
/// smaller pool. [`Self::state_dimension`] remains the global state-vector
/// length used by the cross-cutting checkpoint and boundary paths, equal to
/// every pool's dimension when all stages enable all dimensions (the default).
#[derive(Debug, Clone)]
pub struct FutureCostFunction {
    /// One cut pool per stage, indexed 0-based.
    pub pools: Vec<CutPool>,

    /// Global state-vector length (`StateLayout::n_state`); a pool's own
    /// `state_dimension` may be smaller.
    pub state_dimension: usize,

    /// Forward passes per training iteration. Immutable after construction.
    pub forward_passes: u32,
}

impl FutureCostFunction {
    /// Construct a `FutureCostFunction` with pre-allocated pools, one per stage:
    ///
    /// ```text
    /// capacity[i] = warm_start_counts[i] + max_iterations * forward_passes
    /// ```
    ///
    /// # Parameters
    ///
    /// - `warm_start_counts`: per-stage warm-start cut counts; length must equal
    ///   `num_stages`. Pass `&vec![0; num_stages]` for the no-warm-start case.
    ///
    /// # Example
    ///
    /// ```rust
    /// use cobre_sddp::cut::fcf::FutureCostFunction;
    ///
    /// let fcf = FutureCostFunction::new(5, 9, 10, 100, &[0; 5]);
    /// assert_eq!(fcf.pools.len(), 5);
    /// // capacity = 0 + 100 * 10 = 1000
    /// assert_eq!(fcf.pools[0].capacity, 1000);
    /// ```
    #[must_use]
    pub fn new(
        num_stages: usize,
        state_dimension: usize,
        forward_passes: u32,
        max_iterations: u64,
        warm_start_counts: &[u32],
    ) -> Self {
        Self::new_per_stage(
            &vec![state_dimension; num_stages],
            state_dimension,
            forward_passes,
            max_iterations,
            warm_start_counts,
        )
    }

    /// Construct a `FutureCostFunction` sizing each stage's [`CutPool`] by its own
    /// cut state dimension, for the per-stage `state_config` projection.
    ///
    /// `pool_state_dimensions[t]` is the dimension of pool `t` (the count from
    /// `CutStateProjection(stages[t+1].state_config)` for non-terminal pools; the
    /// global `n_state` for the terminal pool). `global_state_dimension` is
    /// `StateLayout::n_state` — the value stored in [`Self::state_dimension`] and
    /// used by the checkpoint/boundary paths, independent of any reduced pool.
    ///
    /// # Parameters
    ///
    /// - `warm_start_counts`: per-stage warm-start cut counts; length must equal
    ///   `pool_state_dimensions.len()`. Pass `&vec![0; n]` for no warm-start.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `warm_start_counts.len() != pool_state_dimensions.len()`.
    #[must_use]
    pub fn new_per_stage(
        pool_state_dimensions: &[usize],
        global_state_dimension: usize,
        forward_passes: u32,
        max_iterations: u64,
        warm_start_counts: &[u32],
    ) -> Self {
        debug_assert_eq!(
            warm_start_counts.len(),
            pool_state_dimensions.len(),
            "warm_start_counts.len() ({}) != pool_state_dimensions.len() ({})",
            warm_start_counts.len(),
            pool_state_dimensions.len()
        );

        // u64 arithmetic guards the capacity product against overflow; the cast
        // to usize cannot truncate (capacity is memory-bounded on 64-bit targets).
        #[allow(clippy::cast_possible_truncation)]
        let pools = warm_start_counts
            .iter()
            .zip(pool_state_dimensions)
            .map(|(&wsc, &pool_dim)| {
                let capacity =
                    (u64::from(wsc) + max_iterations * u64::from(forward_passes)) as usize;
                CutPool::new(capacity, pool_dim, forward_passes, wsc)
            })
            .collect();

        Self {
            pools,
            state_dimension: global_state_dimension,
            forward_passes,
        }
    }

    /// Reconstruct a `FutureCostFunction` from deserialized policy checkpoint data
    /// via [`CutPool::from_deserialized`] (tightly sized, no training capacity).
    ///
    /// `_proof` is compile-time evidence from
    /// [`validate_policy_load`](crate::validate_policy_load) that
    /// `stage_results` already passed the [`FullFcf`] compatibility check; it is
    /// never read at runtime.
    ///
    /// # Errors
    ///
    /// [`SddpError::Validation`] if `stage_results` is empty or `state_dimension`
    /// is inconsistent across stages.
    ///
    /// [`SddpError::Validation`]: crate::SddpError::Validation
    pub fn from_deserialized(
        _proof: &PolicyLoadProof<FullFcf>,
        stage_results: &[StageCutsReadResult],
    ) -> Result<Self, SddpError> {
        let state_dimension =
            validate_consistent_state_dimension("from_deserialized", stage_results)?;

        let pools = stage_results
            .iter()
            .map(|sr| CutPool::from_deserialized(state_dimension, &sr.cuts))
            .collect();

        Ok(Self {
            pools,
            state_dimension,
            forward_passes: 0,
        })
    }

    /// Build an FCF with warm-start cuts plus training capacity via
    /// [`CutPool::new_with_warm_start`].
    ///
    /// `_proof` is compile-time evidence from
    /// [`validate_policy_load`](crate::validate_policy_load) that
    /// `stage_results` already passed the [`FullFcf`] compatibility check; it is
    /// never read at runtime. Omitting it, or handing a
    /// [`BoundaryInjection`](crate::BoundaryInjection)-typed proof instead,
    /// fails to compile.
    ///
    /// ```compile_fail
    /// use cobre_sddp::FutureCostFunction;
    ///
    /// fn call_without_proof(stage_results: &[cobre_io::StageCutsReadResult]) {
    ///     let _ = FutureCostFunction::new_with_warm_start(stage_results, 4, 10);
    /// }
    /// ```
    ///
    /// ```compile_fail
    /// use cobre_sddp::{BoundaryInjection, FutureCostFunction, PolicyLoadProof};
    ///
    /// fn call_with_wrong_kind(
    ///     proof: &PolicyLoadProof<BoundaryInjection>,
    ///     stage_results: &[cobre_io::StageCutsReadResult],
    /// ) {
    ///     let _ = FutureCostFunction::new_with_warm_start(proof, stage_results, 4, 10);
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// [`SddpError::Validation`] if `stage_results` is empty or `state_dimension`
    /// is inconsistent across stages.
    ///
    /// [`SddpError::Validation`]: crate::SddpError::Validation
    pub fn new_with_warm_start(
        _proof: &PolicyLoadProof<FullFcf>,
        stage_results: &[StageCutsReadResult],
        forward_passes: u32,
        max_iterations: u64,
    ) -> Result<Self, SddpError> {
        let state_dimension =
            validate_consistent_state_dimension("new_with_warm_start", stage_results)?;

        let pools = stage_results
            .iter()
            .map(|sr| {
                CutPool::new_with_warm_start(
                    state_dimension,
                    forward_passes,
                    max_iterations,
                    &sr.cuts,
                )
            })
            .collect();

        Ok(Self {
            pools,
            state_dimension,
            forward_passes,
        })
    }

    /// Insert a Benders cut at the given (0-based) stage. `coefficients` is the
    /// state gradient; length must be `state_dimension`.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `stage >= pools.len()` or if `coefficients.len() !=
    /// state_dimension`.
    pub fn add_cut(
        &mut self,
        stage: usize,
        iteration: u64,
        forward_pass_index: u32,
        intercept: f64,
        coefficients: &[f64],
    ) {
        debug_assert!(
            stage < self.pools.len(),
            "stage index {stage} is out of bounds (num_stages = {})",
            self.pools.len()
        );
        self.pools[stage].add_cut(iteration, forward_pass_index, intercept, coefficients);
    }

    /// Iterate over active cuts at the given (0-based) stage as
    /// `(slot_index, intercept, coefficient_slice)`, for LP construction.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `stage >= pools.len()`.
    pub fn active_cuts(&self, stage: usize) -> impl Iterator<Item = (usize, f64, &[f64])> {
        debug_assert!(
            stage < self.pools.len(),
            "stage index {stage} is out of bounds (num_stages = {})",
            self.pools.len()
        );
        self.pools[stage].active_cuts()
    }

    /// Evaluate the FCF at `values` for the given (0-based) stage: the max over
    /// active cuts, or [`f64::NEG_INFINITY`] when the stage has none. `values`
    /// length must be `state_dimension`.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `stage >= pools.len()` or if `values.len() !=
    /// state_dimension`.
    #[must_use]
    pub fn evaluate_at_state(&self, stage: usize, values: &[f64]) -> f64 {
        debug_assert!(
            stage < self.pools.len(),
            "stage index {stage} is out of bounds (num_stages = {})",
            self.pools.len()
        );
        self.pools[stage].evaluate_at_state(values)
    }

    /// Return the total active-cut count across all stages.
    #[must_use]
    pub fn total_active_cuts(&self) -> usize {
        self.pools.iter().map(CutPool::active_count).sum()
    }

    /// Return the total cuts ever generated across all stages (sums
    /// `generated_count`), excluding reserved-but-unwritten slots — the true
    /// policy-row count.
    #[must_use]
    pub fn total_generated_cuts(&self) -> usize {
        self.pools.iter().map(|p| p.generated_count).sum()
    }

    /// Propagate [`CutPool::set_iteration_base`] to every pool. Call once before
    /// training with `start_iteration + 1` for dense slot packing.
    pub fn set_iteration_base(&mut self, iteration_base: u64) {
        for pool in &mut self.pools {
            pool.set_iteration_base(iteration_base);
        }
    }

    /// Deactivate the cuts at `indices` for the given (0-based) stage.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `stage >= pools.len()`.
    pub fn deactivate(&mut self, stage: usize, indices: &[u32]) {
        debug_assert!(
            stage < self.pools.len(),
            "stage index {stage} is out of bounds (num_stages = {})",
            self.pools.len()
        );
        self.pools[stage].deactivate(indices);
    }

    /// Toggle a single slot's activity flag at the given (0-based) stage via
    /// [`CutPool::set_active`].
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `stage >= pools.len()` or if `slot >= pools[stage].populated()`.
    ///
    /// [`CutPool::set_active`]: super::pool::CutPool::set_active
    pub fn set_active(&mut self, stage: usize, slot: u32, active: bool) {
        debug_assert!(
            stage < self.pools.len(),
            "stage index {stage} is out of bounds (num_stages = {})",
            self.pools.len()
        );
        self.pools[stage].set_active(slot, active);
    }

    /// Return the LP-row count (populated slots, independent of activity) for the
    /// given (0-based) stage via [`CutPool::cuts_in_lp`].
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if `stage >= pools.len()`.
    ///
    /// [`CutPool::cuts_in_lp`]: super::pool::CutPool::cuts_in_lp
    #[must_use]
    pub fn cuts_in_lp(&self, stage: usize) -> usize {
        debug_assert!(
            stage < self.pools.len(),
            "stage index {stage} is out of bounds (num_stages = {})",
            self.pools.len()
        );
        self.pools[stage].cuts_in_lp()
    }

    /// One [`SparsityReport`] per stage, in stage order.
    ///
    /// [`SparsityReport`]: super::pool::SparsityReport
    #[must_use]
    pub fn sparsity_reports(&self) -> Vec<super::pool::SparsityReport> {
        self.pools.iter().map(CutPool::sparsity_report).collect()
    }
}

fn validate_consistent_state_dimension(
    context: &str,
    stage_results: &[StageCutsReadResult],
) -> Result<usize, SddpError> {
    if stage_results.is_empty() {
        return Err(Validation(format!("{context}: stage_results is empty")));
    }

    let state_dimension = stage_results[0].state_dimension as usize;
    for sr in &stage_results[1..] {
        if sr.state_dimension as usize != state_dimension {
            return Err(Validation(format!(
                "{context}: inconsistent state_dimension: stage {} has {}, \
                 expected {} (from stage {})",
                sr.stage_id, sr.state_dimension, state_dimension, stage_results[0].stage_id
            )));
        }
    }

    Ok(state_dimension)
}

#[cfg(test)]
mod tests {
    use super::FutureCostFunction;

    #[test]
    fn new_creates_correct_number_of_pools() {
        let fcf = FutureCostFunction::new(5, 9, 10, 100, &[0; 5]);
        assert_eq!(fcf.pools.len(), 5);
    }

    #[test]
    fn new_each_pool_has_correct_capacity_no_warmstart() {
        // capacity = 0 + 100 * 10 = 1000
        let fcf = FutureCostFunction::new(5, 9, 10, 100, &[0; 5]);
        for pool in &fcf.pools {
            assert_eq!(pool.capacity, 1000);
            assert_eq!(pool.state_dimension, 9);
            assert_eq!(pool.forward_passes, 10);
            assert_eq!(pool.warm_start_count, 0);
        }
    }

    #[test]
    fn new_each_pool_has_correct_capacity_with_warmstart() {
        // capacity = 5 + 100 * 10 = 1005
        let fcf = FutureCostFunction::new(3, 4, 10, 100, &[5; 3]);
        for pool in &fcf.pools {
            assert_eq!(pool.capacity, 1005);
            assert_eq!(pool.warm_start_count, 5);
        }
    }

    #[test]
    fn new_all_pools_start_with_zero_active_cuts() {
        let fcf = FutureCostFunction::new(4, 3, 5, 20, &[0; 4]);
        assert_eq!(fcf.total_active_cuts(), 0);
    }

    #[test]
    fn new_zero_stages_is_valid() {
        let fcf = FutureCostFunction::new(0, 4, 5, 10, &[]);
        assert_eq!(fcf.pools.len(), 0);
        assert_eq!(fcf.total_active_cuts(), 0);
    }

    #[test]
    fn new_non_uniform_warm_start_counts_per_stage_capacity() {
        // warm_start_counts = [5, 3, 0], max_iterations = 10, forward_passes = 2
        // capacity[0] = 5 + 10*2 = 25
        // capacity[1] = 3 + 10*2 = 23
        // capacity[2] = 0 + 10*2 = 20
        let fcf = FutureCostFunction::new(3, 4, 2, 10, &[5, 3, 0]);
        assert_eq!(fcf.pools[0].capacity, 25);
        assert_eq!(fcf.pools[0].warm_start_count, 5);
        assert_eq!(fcf.pools[1].capacity, 23);
        assert_eq!(fcf.pools[1].warm_start_count, 3);
        assert_eq!(fcf.pools[2].capacity, 20);
        assert_eq!(fcf.pools[2].warm_start_count, 0);
    }

    #[test]
    fn new_uniform_zero_counts_matches_old_scalar_zero_behavior() {
        let fcf = FutureCostFunction::new(3, 4, 2, 10, &[0, 0, 0]);
        for pool in &fcf.pools {
            // capacity = 0 + 10*2 = 20
            assert_eq!(pool.capacity, 20);
            assert_eq!(pool.warm_start_count, 0);
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "warm_start_counts.len()")]
    fn new_mismatched_length_panics_in_debug() {
        // Length 2 != num_stages 3: should trigger debug_assert_eq!
        let _ = FutureCostFunction::new(3, 4, 2, 10, &[0, 0]);
    }

    #[test]
    fn add_cut_and_active_cuts_round_trip_at_specific_stage() {
        let mut fcf = FutureCostFunction::new(5, 2, 1, 10, &[0; 5]);
        let coeffs = [3.0, 7.0];
        fcf.add_cut(2, 0, 0, 42.0, &coeffs);

        let active: Vec<_> = fcf.active_cuts(2).collect();
        assert_eq!(active.len(), 1);
        let (_, intercept, c) = active[0];
        assert_eq!(intercept, 42.0);
        assert_eq!(c, &[3.0, 7.0]);
    }

    #[test]
    fn active_cuts_at_other_stage_returns_empty() {
        let mut fcf = FutureCostFunction::new(5, 2, 1, 10, &[0; 5]);
        fcf.add_cut(2, 0, 0, 42.0, &[1.0, 2.0]);

        let active: Vec<_> = fcf.active_cuts(3).collect();
        assert!(active.is_empty());
    }

    #[test]
    fn add_cut_multiple_stages_are_independent() {
        let mut fcf = FutureCostFunction::new(4, 1, 1, 10, &[0; 4]);
        fcf.add_cut(0, 0, 0, 1.0, &[1.0]);
        fcf.add_cut(1, 0, 0, 2.0, &[2.0]);
        fcf.add_cut(3, 0, 0, 4.0, &[4.0]);

        assert_eq!(fcf.active_cuts(0).count(), 1);
        assert_eq!(fcf.active_cuts(1).count(), 1);
        assert_eq!(fcf.active_cuts(2).count(), 0);
        assert_eq!(fcf.active_cuts(3).count(), 1);
    }

    #[test]
    fn evaluate_at_state_delegates_to_correct_pool() {
        let mut fcf = FutureCostFunction::new(3, 2, 1, 10, &[0; 3]);
        // stage 1: cut with intercept=10, coeffs=[1,0]
        fcf.add_cut(1, 0, 0, 10.0, &[1.0, 0.0]);
        // stage 2: cut with intercept=5, coeffs=[0,2]
        fcf.add_cut(2, 0, 0, 5.0, &[0.0, 2.0]);

        // stage 1: 10 + 1*3 + 0*4 = 13
        assert_eq!(fcf.evaluate_at_state(1, &[3.0, 4.0]), 13.0);
        // stage 2: 5 + 0*3 + 2*4 = 13
        assert_eq!(fcf.evaluate_at_state(2, &[3.0, 4.0]), 13.0);
        // stage 0: no cuts → NEG_INFINITY
        assert_eq!(fcf.evaluate_at_state(0, &[3.0, 4.0]), f64::NEG_INFINITY);
    }

    #[test]
    fn total_active_cuts_sums_across_stages() {
        let mut fcf = FutureCostFunction::new(4, 1, 1, 20, &[0; 4]);
        fcf.add_cut(0, 0, 0, 1.0, &[1.0]);
        fcf.add_cut(1, 0, 0, 2.0, &[2.0]);
        fcf.add_cut(1, 1, 0, 3.0, &[3.0]);
        fcf.add_cut(3, 0, 0, 4.0, &[4.0]);

        // stage 0: 1, stage 1: 2, stage 2: 0, stage 3: 1 → total = 4
        assert_eq!(fcf.total_active_cuts(), 4);
    }

    #[test]
    fn total_active_cuts_reflects_deactivation() {
        let mut fcf = FutureCostFunction::new(2, 1, 1, 10, &[0; 2]);
        fcf.add_cut(0, 0, 0, 1.0, &[1.0]); // slot 0
        fcf.add_cut(0, 1, 0, 2.0, &[2.0]); // slot 1
        fcf.add_cut(1, 0, 0, 3.0, &[3.0]); // slot 0

        assert_eq!(fcf.total_active_cuts(), 3);
        fcf.deactivate(0, &[0]);
        assert_eq!(fcf.total_active_cuts(), 2);
    }

    #[test]
    fn set_iteration_base_propagates_and_packs_densely() {
        let mut fcf = FutureCostFunction::new(3, 1, 2, 10, &[0; 3]);
        fcf.set_iteration_base(1);
        for pool in &fcf.pools {
            assert_eq!(pool.iteration_base, 1);
        }
        // With base 1, iteration 1 maps to slot 0: no reserved leading block
        // (base 0 would place these at slots 2,3 with populated_count 4).
        fcf.add_cut(0, 1, 0, 1.0, &[1.0]);
        fcf.add_cut(0, 1, 1, 2.0, &[1.0]);
        assert_eq!(fcf.pools[0].populated(), 2);
        assert_eq!(fcf.pools[0].generated_count, 2);
    }

    #[test]
    fn deactivate_delegates_to_correct_pool() {
        let mut fcf = FutureCostFunction::new(3, 1, 1, 10, &[0; 3]);
        fcf.add_cut(1, 0, 0, 10.0, &[1.0]); // slot 0 of pool[1]
        fcf.add_cut(1, 1, 0, 20.0, &[2.0]); // slot 1 of pool[1]
        fcf.add_cut(2, 0, 0, 30.0, &[3.0]); // slot 0 of pool[2]

        fcf.deactivate(1, &[0]);

        // pool[1] should now have 1 active cut
        assert_eq!(fcf.active_cuts(1).count(), 1);
        // pool[2] should still have 1 active cut
        assert_eq!(fcf.active_cuts(2).count(), 1);
    }

    #[test]
    fn ac_new_5_stages_pools_len_is_5() {
        let fcf = FutureCostFunction::new(5, 9, 10, 100, &[0; 5]);
        assert_eq!(fcf.pools.len(), 5);
    }

    #[test]
    fn ac_active_cuts_at_stage_with_cut_yields_it() {
        let mut fcf = FutureCostFunction::new(5, 3, 1, 10, &[0; 5]);
        let coeffs = [1.0, 2.0, 3.0];
        fcf.add_cut(2, 0, 0, 99.0, &coeffs);

        let active: Vec<_> = fcf.active_cuts(2).collect();
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn ac_active_cuts_at_different_stage_yields_none() {
        let mut fcf = FutureCostFunction::new(5, 3, 1, 10, &[0; 5]);
        fcf.add_cut(2, 0, 0, 99.0, &[1.0, 2.0, 3.0]);

        let active: Vec<_> = fcf.active_cuts(3).collect();
        assert!(active.is_empty());
    }

    #[test]
    fn ac_total_active_cuts_is_sum_across_stages() {
        let mut fcf = FutureCostFunction::new(5, 1, 1, 10, &[0; 5]);
        fcf.add_cut(0, 0, 0, 1.0, &[1.0]);
        fcf.add_cut(1, 0, 0, 2.0, &[2.0]);
        fcf.add_cut(1, 1, 0, 3.0, &[3.0]);
        fcf.add_cut(4, 0, 0, 4.0, &[4.0]);

        assert_eq!(fcf.total_active_cuts(), 4);
    }

    #[test]
    fn fcf_derives_debug_and_clone() {
        let mut fcf = FutureCostFunction::new(2, 2, 1, 5, &[0; 2]);
        fcf.add_cut(0, 0, 0, 7.0, &[1.0, 2.0]);

        let cloned = fcf.clone();
        assert_eq!(cloned.total_active_cuts(), 1);
        assert_eq!(cloned.evaluate_at_state(0, &[0.0, 0.0]), 7.0);

        let debug_str = format!("{fcf:?}");
        assert!(!debug_str.is_empty());
    }

    // ── from_deserialized tests ──────────────────────────────────────────

    fn make_record(
        intercept: f64,
        coefficients: Vec<f64>,
        is_active: bool,
    ) -> cobre_io::OwnedPolicyCutRecord {
        cobre_io::OwnedPolicyCutRecord {
            cut_id: 0,
            slot_index: 0,
            iteration: 0,
            forward_pass_index: 0,
            intercept,
            coefficients,
            is_active,
        }
    }

    fn make_stage(
        stage_id: u32,
        state_dimension: u32,
        cuts: Vec<cobre_io::OwnedPolicyCutRecord>,
    ) -> cobre_io::StageCutsReadResult {
        let populated_count = u32::try_from(cuts.len()).expect("cuts count fits in u32");
        cobre_io::StageCutsReadResult {
            stage_id,
            state_dimension,
            capacity: populated_count,
            warm_start_count: 0,
            populated_count,
            cuts,
            entity_manifest: Vec::new(),
        }
    }

    #[test]
    fn from_deserialized_empty_input_returns_err() {
        let proof = crate::test_support::trivial_full_fcf_proof(0, 0);
        let result = FutureCostFunction::from_deserialized(&proof, &[]);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("empty"), "{msg}");
    }

    #[test]
    fn from_deserialized_inconsistent_dimensions_returns_err() {
        let stages = vec![
            make_stage(0, 2, vec![make_record(1.0, vec![1.0, 0.0], true)]),
            make_stage(1, 3, vec![make_record(2.0, vec![1.0, 0.0, 0.0], true)]),
        ];
        let proof = crate::test_support::trivial_full_fcf_proof(2, 2);
        let result = FutureCostFunction::from_deserialized(&proof, &stages);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("inconsistent"), "{msg}");
    }

    #[test]
    fn from_deserialized_preserves_active_flags() {
        let stages = vec![make_stage(
            0,
            2,
            vec![
                make_record(1.0, vec![1.0, 0.0], true),
                make_record(2.0, vec![0.0, 1.0], false), // inactive
                make_record(3.0, vec![1.0, 1.0], true),
            ],
        )];

        let proof = crate::test_support::trivial_full_fcf_proof(2, 1);
        let fcf = FutureCostFunction::from_deserialized(&proof, &stages).unwrap();
        assert_eq!(fcf.pools.len(), 1);
        assert_eq!(fcf.total_active_cuts(), 2);
        assert_eq!(fcf.pools[0].populated(), 3);
    }

    #[test]
    fn from_deserialized_evaluate_at_state_matches_original() {
        // Build original FCF with known cuts, then reconstruct via deserialized.
        let mut original = FutureCostFunction::new(2, 2, 1, 10, &[0; 2]);
        original.add_cut(0, 0, 0, 10.0, &[1.0, 0.0]);
        original.add_cut(0, 1, 0, 5.0, &[0.0, 2.0]);
        original.add_cut(1, 0, 0, 3.0, &[1.0, 1.0]);

        let state = [3.0, 4.0];
        let orig_val_s0 = original.evaluate_at_state(0, &state);
        let orig_val_s1 = original.evaluate_at_state(1, &state);

        // Construct deserialized data matching the original cuts.
        let stages = vec![
            make_stage(
                0,
                2,
                vec![
                    make_record(10.0, vec![1.0, 0.0], true),
                    make_record(5.0, vec![0.0, 2.0], true),
                ],
            ),
            make_stage(1, 2, vec![make_record(3.0, vec![1.0, 1.0], true)]),
        ];

        let proof = crate::test_support::trivial_full_fcf_proof(2, 2);
        let reconstructed = FutureCostFunction::from_deserialized(&proof, &stages).unwrap();
        assert_eq!(reconstructed.evaluate_at_state(0, &state), orig_val_s0);
        assert_eq!(reconstructed.evaluate_at_state(1, &state), orig_val_s1);
    }

    #[test]
    fn from_deserialized_empty_stage_is_valid() {
        let stages = vec![
            make_stage(0, 2, vec![make_record(1.0, vec![1.0, 0.0], true)]),
            make_stage(1, 2, vec![]), // empty stage
        ];

        let proof = crate::test_support::trivial_full_fcf_proof(2, 2);
        let fcf = FutureCostFunction::from_deserialized(&proof, &stages).unwrap();
        assert_eq!(fcf.pools.len(), 2);
        assert_eq!(fcf.pools[1].capacity, 0);
        assert_eq!(fcf.pools[1].active_count(), 0);
        assert_eq!(fcf.evaluate_at_state(1, &[1.0, 1.0]), f64::NEG_INFINITY);
    }

    #[test]
    fn from_deserialized_single_cut_stage() {
        let stages = vec![make_stage(
            0,
            3,
            vec![make_record(7.0, vec![1.0, 2.0, 3.0], true)],
        )];

        let proof = crate::test_support::trivial_full_fcf_proof(3, 1);
        let fcf = FutureCostFunction::from_deserialized(&proof, &stages).unwrap();
        assert_eq!(fcf.state_dimension, 3);
        assert_eq!(fcf.total_active_cuts(), 1);
        // 7 + 1*1 + 2*2 + 3*3 = 7 + 1 + 4 + 9 = 21
        assert_eq!(fcf.evaluate_at_state(0, &[1.0, 2.0, 3.0]), 21.0);
    }

    // ── new_with_warm_start tests ────────────────────────────────────────

    #[test]
    fn warm_start_capacity_includes_training_slots() {
        let stages = vec![make_stage(
            0,
            2,
            vec![
                make_record(1.0, vec![1.0, 0.0], true),
                make_record(2.0, vec![0.0, 1.0], true),
            ],
        )];

        let proof = crate::test_support::trivial_full_fcf_proof(2, 1);
        let fcf = FutureCostFunction::new_with_warm_start(&proof, &stages, 4, 10).unwrap();
        assert_eq!(fcf.pools.len(), 1);
        // capacity = 2 warm-start + 10*4 training = 42
        assert_eq!(fcf.pools[0].capacity, 42);
        assert_eq!(fcf.pools[0].warm_start_count, 2);
        assert_eq!(fcf.pools[0].forward_passes, 4);
        assert_eq!(fcf.pools[0].populated(), 2);
        assert_eq!(fcf.total_active_cuts(), 2);
    }

    #[test]
    fn warm_start_training_cuts_at_correct_offset() {
        let stages = vec![make_stage(0, 1, vec![make_record(10.0, vec![1.0], true)])];

        let proof = crate::test_support::trivial_full_fcf_proof(1, 1);
        let mut fcf = FutureCostFunction::new_with_warm_start(&proof, &stages, 2, 5).unwrap();
        // warm_start_count = 1, forward_passes = 2
        // Training cut at iteration=0, fwd_idx=0: slot = 1 + 0*2 + 0 = 1
        fcf.add_cut(0, 0, 0, 20.0, &[2.0]);
        // Training cut at iteration=0, fwd_idx=1: slot = 1 + 0*2 + 1 = 2
        fcf.add_cut(0, 0, 1, 30.0, &[3.0]);

        assert_eq!(fcf.total_active_cuts(), 3);
        assert_eq!(fcf.pools[0].populated(), 3);
        // Warm-start cut at slot 0
        assert_eq!(fcf.pools[0].intercept(0), 10.0);
        // Training cuts at slots 1 and 2
        assert_eq!(fcf.pools[0].intercept(1), 20.0);
        assert_eq!(fcf.pools[0].intercept(2), 30.0);
    }

    #[test]
    fn warm_start_empty_stage_has_training_capacity() {
        let stages = vec![
            make_stage(0, 2, vec![make_record(1.0, vec![1.0, 0.0], true)]),
            make_stage(1, 2, vec![]),
        ];

        let proof = crate::test_support::trivial_full_fcf_proof(2, 2);
        let fcf = FutureCostFunction::new_with_warm_start(&proof, &stages, 3, 5).unwrap();
        // Stage 0: warm_start=1, capacity=1+15=16
        assert_eq!(fcf.pools[0].capacity, 16);
        assert_eq!(fcf.pools[0].warm_start_count, 1);
        // Stage 1: warm_start=0, capacity=0+15=15
        assert_eq!(fcf.pools[1].capacity, 15);
        assert_eq!(fcf.pools[1].warm_start_count, 0);
    }

    #[test]
    fn warm_start_preserves_inactive_flags() {
        let stages = vec![make_stage(
            0,
            2,
            vec![
                make_record(1.0, vec![1.0, 0.0], true),
                make_record(2.0, vec![0.0, 1.0], false), // inactive
            ],
        )];

        let proof = crate::test_support::trivial_full_fcf_proof(2, 1);
        let fcf = FutureCostFunction::new_with_warm_start(&proof, &stages, 1, 5).unwrap();
        assert_eq!(fcf.pools[0].warm_start_count, 2);
        assert_eq!(fcf.total_active_cuts(), 1); // only the active one
    }

    // ── set_active tests ────────────────────────────────────────────────────

    #[test]
    fn fcf_set_active_delegates_to_pool() {
        let mut fcf = FutureCostFunction::new(3, 1, 1, 10, &[0; 3]);
        // Add 3 cuts to stage 1: slots 0, 1, 2
        fcf.add_cut(1, 0, 0, 10.0, &[1.0]);
        fcf.add_cut(1, 1, 0, 20.0, &[2.0]);
        fcf.add_cut(1, 2, 0, 30.0, &[3.0]);
        let prior = fcf.total_active_cuts();

        fcf.set_active(1, 0, false);

        assert!(!fcf.pools[1].is_active(0));
        assert_eq!(fcf.total_active_cuts(), prior - 1);
    }
}
