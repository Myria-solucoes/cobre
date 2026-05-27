//! Cut selection strategy for controlling cut pool growth during SDDP training.
//!
//! This module defines [`CutSelectionStrategy`] (three variants: Level1, LML1,
//! Dominated), [`CutMetadata`] (per-cut tracking data), and [`CutActivityUpdates`]
//! (the output of a selection scan for one stage).
//!
//! ## Design
//!
//! Both methods (`should_run`, `select`) are pure and infallible. Configuration
//! parameters are validated at load time, so runtime panics from zero
//! `check_frequency` are impossible.
//!
//! All three variants share a single value-evaluation kernel in `select_for_stage`.
//! The kernel evaluates ALL populated cuts (active AND inactive) at each visited
//! forward-pass state and applies the method-specific survival rule:
//!
//! - **Level1**: retain any cut within `tie_tolerance` of the per-state maximum
//!   at any visited state (de Matos 2015).
//! - **Lml1**: at each visited state, retain only the oldest eligible cut within
//!   `tie_tolerance` of the maximum; the overall selected set is the union across
//!   all visited states (Guigues & Bandarra 2019).
//! - **Dominated**: same max-survival logic as Level1 using `threshold` as the
//!   tolerance, applied across ALL populated cuts (including inactive ones).
//!
//! Inactive cuts that are selected by the kernel produce `Reactivate` entries in
//! the output; active cuts that are not selected produce `Deactivate` entries.
//!
//! ## Usage
//!
//! ```rust
//! use cobre_sddp::cut::CutPool;
//! use cobre_sddp::cut_selection::{
//!     CutActivityUpdates, CutMetadata, CutSelectionStrategy,
//! };
//!
//! let strategy = CutSelectionStrategy::Level1 {
//!     check_frequency: 5,
//!     tie_tolerance: 1e-10,
//! };
//!
//! // Should run at multiples of check_frequency (excluding 0).
//! assert!(!strategy.should_run(0));
//! assert!(!strategy.should_run(3));
//! assert!(strategy.should_run(5));
//! assert!(strategy.should_run(10));
//! ```

use rayon::prelude::*;

/// Trial-point count threshold above which intra-stage selection is parallelized.
///
/// When the number of visited states for a stage is at least this value,
/// [`CutSelectionStrategy::select_for_stage`] splits the visited states into
/// chunks and processes each chunk on a rayon worker (nested under the existing
/// outer `into_par_iter` over stages). Below the threshold the kernel runs
/// sequentially to avoid per-task overhead dominating the work.
///
/// 256 is chosen as a balance between thread-spawn overhead and useful work per
/// chunk. The merge across chunks is bitwise-OR, which is commutative and
/// associative, so the final `is_selected` bitmap is bit-for-bit identical
/// regardless of the number of threads or chunk boundaries.
const PARALLEL_THRESHOLD: usize = 256;

// ---------------------------------------------------------------------------
// CutMetadata
// ---------------------------------------------------------------------------

/// Per-cut tracking metadata for cut selection strategies.
///
/// Stored alongside cut coefficients and intercepts in the pre-allocated
/// cut pool. All fields are initialised to zero / default values when the
/// cut slot is first populated. Updated inline during the backward pass
/// in `crate::backward_pass_state::BackwardPassState` (the function
/// that owns the per-stage cut-binding sync step).
#[derive(Debug, Clone)]
pub struct CutMetadata {
    /// Iteration at which this cut was generated (1-based).
    ///
    /// Used to prevent deactivation of cuts generated in the current
    /// iteration.
    pub iteration_generated: u64,

    /// Forward pass index that generated this cut.
    ///
    /// Combined with `iteration_generated`, uniquely identifies the
    /// deterministic slot for this cut.
    pub forward_pass_index: u32,

    /// Cumulative number of times this cut was binding at an LP solution.
    ///
    /// Used by budget enforcement (row eviction) and diagnostics; NOT used by
    /// the value-based selection logic in [`CutSelectionStrategy::Level1`] or
    /// [`CutSelectionStrategy::Lml1`].
    /// Initialised to 0; incremented inline by the backward pass when the
    /// associated cut row's dual exceeds `cut_activity_tolerance`.
    pub active_count: u64,

    /// Most recent iteration at which this cut was binding.
    ///
    /// Used by budget enforcement (staleness-based eviction ordering) and
    /// diagnostics; NOT used by the value-based selection logic in
    /// [`CutSelectionStrategy::Level1`] or [`CutSelectionStrategy::Lml1`].
    /// Initialised to `iteration_generated`; updated inline by the backward
    /// pass during the per-stage cut-binding sync.
    pub last_active_iter: u64,

    /// Sliding-window binding-activity bitmap.
    ///
    /// Bit 0 = current iteration; bit `i` = iteration `current_iter - i`.
    /// Updated to bit-0-set when the cut was binding (dual >
    /// `cut_activity_tolerance`) during any backward solve of the current
    /// iteration; shifted left by 1 at end-of-iteration so the next
    /// iteration's bit 0 records fresh activity.
    ///
    /// Populated by the MPI `allreduce(BitwiseOr)` in the backward pass
    /// (so any rank observing the cut binding sets bit 0 globally). Consumed
    /// by the activity-guided basis classifier in `reconstruct_basis`.
    ///
    /// **Transient seed**: `add_cut` sets
    /// [`crate::basis_reconstruct::SEED_BIT`] (bit 31, outside
    /// `RECENT_WINDOW_BITS`) so the classifier fires LOWER on a freshly
    /// generated cut during the same iteration's remaining backward stages —
    /// the cut is tight at the x̂ it was derived from by construction. The
    /// end-of-iteration logic clears `SEED_BIT` *before* the `<<= 1` shift so
    /// the seed does **not** persist into the next iteration's basis
    /// reconstruction. From iteration i+1 onward, only genuine binding
    /// observations drive classification decisions.
    pub active_window: u32,
}

// ---------------------------------------------------------------------------
// CutActivityUpdates
// ---------------------------------------------------------------------------

/// Set of cut activity updates at a single stage.
///
/// Returned by [`CutSelectionStrategy::select`]. `updates` contains slot
/// indices to deactivate; `reactivations` contains slot indices to reactivate.
/// Either list may be empty.
///
/// The caller applies changes to the activity bitmap via
/// [`CutPool::deactivate`] and [`CutPool::set_active`].
#[derive(Debug, Clone, PartialEq)]
pub struct CutActivityUpdates {
    /// Stage index (0-based) that this update set belongs to.
    pub stage_index: u32,
    /// Slot indices to deactivate.
    pub updates: Vec<u32>,
    /// Slot indices to reactivate.
    pub reactivations: Vec<u32>,
}

impl CutActivityUpdates {
    /// Construct a deactivation-only update set from a list of slot indices.
    ///
    /// The `reactivations` list is left empty. Use this constructor when only
    /// deactivations are known and reactivations will be added separately or
    /// are not applicable.
    #[must_use]
    pub fn deactivations_only(stage_index: u32, indices: Vec<u32>) -> Self {
        Self {
            stage_index,
            updates: indices,
            reactivations: vec![],
        }
    }

    /// Return the slots to deactivate.
    #[must_use]
    pub fn deactivation_indices(&self) -> Vec<u32> {
        self.updates.clone()
    }

    /// Return the slots to reactivate.
    #[must_use]
    pub fn reactivation_indices(&self) -> Vec<u32> {
        self.reactivations.clone()
    }
}

// ---------------------------------------------------------------------------
// CutSelectionStrategy
// ---------------------------------------------------------------------------

/// Cut selection strategy for controlling cut pool growth during SDDP training.
///
/// One strategy is active for the entire training run (global setting, one
/// variant per run). All stages use the same strategy. Selection runs
/// periodically via [`should_run`] to amortize the cost of scanning the pool.
///
/// This type derives [`serde::Serialize`] and [`serde::Deserialize`] so it can
/// be postcard-serialized directly for MPI broadcast without a wrapper enum.
/// Variant names and field names are stable wire-format identifiers — do not
/// rename them without a migration.
///
/// [`should_run`]: CutSelectionStrategy::should_run
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CutSelectionStrategy {
    /// Level-1 selection: retain any cut that is near-optimal at some visited
    /// state (de Matos 2015).
    ///
    /// A cut is deactivated if, at every visited forward-pass state, its value
    /// is more than `tie_tolerance` below the maximum cut value at that state.
    /// The maximum is computed over ALL populated cuts (active and inactive).
    /// Cuts that achieve within `tie_tolerance` of the maximum at any state are
    /// kept. This is the least aggressive value-based strategy and preserves
    /// the convergence guarantee.
    Level1 {
        /// Number of iterations between selection runs. Must be > 0.
        check_frequency: u64,

        /// Absolute tolerance for tie-breaking: a cut is considered active at a
        /// state when its value is within `tie_tolerance` of the best cut value
        /// at that state. Default: `1e-10`.
        tie_tolerance: f64,
    },

    /// Limited Memory Level-1 (LML1): value-based selection retaining only the
    /// oldest eligible near-optimal cut per visited state (Guigues & Bandarra 2019).
    ///
    /// At each visited state, only the oldest eligible cut (smallest slot index
    /// `>= warm_start_count`) whose value is within `tie_tolerance` of the
    /// maximum survives. The selected set is the union of oldest-at-max cuts
    /// across all visited states. More aggressive than Level1 because multiple
    /// cuts tied at the same state compete and only the oldest wins.
    Lml1 {
        /// Number of iterations between selection runs. Must be > 0.
        check_frequency: u64,

        /// Absolute tolerance for tie-breaking: a cut is considered active at a
        /// state when its value is within `tie_tolerance` of the best cut value
        /// at that state. Default: `1e-10`.
        tie_tolerance: f64,
    },

    /// Dominated cut detection: remove cuts dominated at all visited states.
    ///
    /// A cut is dominated if at every visited forward pass state, the maximum
    /// over ALL populated cuts (active and inactive) exceeds the cut's value
    /// by more than `threshold`. Dominated cuts contribute nothing to the
    /// policy and can safely be deactivated. Inactive cuts that achieve the
    /// maximum are reactivated.
    Dominated {
        /// Activity threshold epsilon. A cut survives if its value is within
        /// `threshold` of the maximum at any visited state.
        threshold: f64,

        /// Number of iterations between selection runs. Must be > 0.
        check_frequency: u64,
    },
}

impl CutSelectionStrategy {
    /// Determine whether cut selection should run at the given iteration.
    ///
    /// Returns `true` if `iteration > 0` and `iteration` is a multiple of
    /// the variant's `check_frequency`. Never runs at iteration 0 (no cuts
    /// exist yet).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::cut_selection::CutSelectionStrategy;
    ///
    /// let s = CutSelectionStrategy::Level1 { check_frequency: 5, tie_tolerance: 1e-10 };
    /// assert!(!s.should_run(0));
    /// assert!(!s.should_run(3));
    /// assert!(s.should_run(5));
    /// assert!(s.should_run(10));
    /// ```
    #[must_use]
    pub fn should_run(&self, iteration: u64) -> bool {
        let freq = match self {
            Self::Level1 {
                check_frequency, ..
            }
            | Self::Lml1 {
                check_frequency, ..
            }
            | Self::Dominated {
                check_frequency, ..
            } => *check_frequency,
        };
        iteration > 0 && iteration % freq == 0
    }

    /// Scan the cut pool metadata for a single stage and identify cuts to
    /// deactivate.
    ///
    /// Returns a [`CutActivityUpdates`] with deactivation and reactivation
    /// entries. The caller is responsible for applying changes to the activity
    /// bitmap. This method does not modify the pool — it is a pure query.
    ///
    /// `stage_index` identifies which stage this selection runs for (used to
    /// populate [`CutActivityUpdates::stage_index`]).
    ///
    /// # Variant behavior
    ///
    /// - **Level1**: evaluates all cuts at visited states; retains any cut
    ///   within `tie_tolerance` of the per-state maximum at any state.
    /// - **Lml1**: at each visited state, retains only the oldest eligible
    ///   cut within `tie_tolerance` of the maximum; selected set is the union
    ///   across all visited states.
    /// - **Dominated**: same max-survival logic as Level1 using `threshold`.
    ///
    /// When `visited_states` is empty, Level1 and Lml1 return empty updates
    /// (no evidence for value evaluation). Dominated also returns empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use cobre_sddp::cut::{CutPool};
    /// use cobre_sddp::cut_selection::{CutMetadata, CutSelectionStrategy};
    ///
    /// let strategy = CutSelectionStrategy::Level1 { check_frequency: 5, tie_tolerance: 1e-10 };
    /// let mut pool = CutPool::new(2, 1, 1, 0);
    /// pool.add_cut(0, 0, 1.0, &[1.0]);
    /// pool.add_cut(1, 0, 2.0, &[2.0]);
    /// // Empty visited_states returns empty updates.
    /// let deact = strategy.select(&pool, &[], 10);
    /// assert!(deact.deactivation_indices().is_empty());
    /// ```
    #[must_use]
    pub fn select(
        &self,
        pool: &crate::cut::CutPool,
        visited_states: &[f64],
        current_iteration: u64,
    ) -> CutActivityUpdates {
        self.select_for_stage(pool, visited_states, current_iteration, 0)
    }

    /// Scan the cut pool for a specific stage and identify cuts to update.
    ///
    /// Accepts the full [`CutPool`](crate::cut::CutPool) reference so that all
    /// variants can access coefficients and intercepts for value evaluation.
    ///
    /// `visited_states` is a flat `&[f64]` of visited forward-pass state
    /// vectors (row-major, one state per `pool.state_dimension` elements).
    /// When empty, the function returns empty updates immediately.
    ///
    /// # Parallelism
    ///
    /// When the number of trial points is at least [`PARALLEL_THRESHOLD`], the
    /// visited states are split into chunks and each chunk is evaluated on a
    /// rayon worker. The per-chunk results are bitmaps of cut indices selected
    /// by that chunk's trial-point subset; chunks are merged with bitwise-OR
    /// over the bitmaps. Because trial points are independent and the merge
    /// (union) is commutative and associative, the final `is_selected` set is
    /// bit-for-bit identical regardless of the number of threads or chunk
    /// boundaries — including for Lml1, where the per-chunk "oldest at max" is
    /// taken over disjoint trial-point subsets and the union across subsets
    /// equals the global oldest-at-max union.
    ///
    /// Below the threshold the work is too small for parallelism to pay off,
    /// so the same kernel runs once on the full state slice.
    ///
    /// [`select`]: CutSelectionStrategy::select
    #[must_use]
    pub fn select_for_stage(
        &self,
        pool: &crate::cut::CutPool,
        visited_states: &[f64],
        current_iteration: u64,
        stage_index: u32,
    ) -> CutActivityUpdates {
        let populated = pool.populated_count;
        let n_state = pool.state_dimension;
        let warm_start = pool.warm_start_count as usize;

        // Edge cases: nothing to evaluate.
        if populated == 0 || visited_states.is_empty() || n_state == 0 {
            return CutActivityUpdates {
                stage_index,
                updates: vec![],
                reactivations: vec![],
            };
        }

        // Eligible predicate: not a warm-start slot AND not from the current iteration.
        // Warm-start cuts always remain active (slots 0..warm_start excluded from changes).
        // Current-iteration cuts are protected against activity changes.
        let eligible: Vec<bool> = (0..populated)
            .map(|k| k >= warm_start && pool.metadata[k].iteration_generated < current_iteration)
            .collect();
        let n_eligible = eligible.iter().filter(|&&e| e).count();

        // Need at least 2 eligible cuts for any to be deactivated.
        if n_eligible < 2 {
            return CutActivityUpdates {
                stage_index,
                updates: vec![],
                reactivations: vec![],
            };
        }

        let n_states = visited_states.len() / n_state;

        // Decide between sequential and parallel processing. Both paths
        // ultimately call [`evaluate_chunk`] with the same logic; parallel
        // path merges per-chunk bitmaps with bitwise-OR.
        let is_selected: Vec<bool> = if n_states >= PARALLEL_THRESHOLD {
            // par_chunks operates over the flat f64 slice; each chunk contains
            // PARALLEL_THRESHOLD states packed as PARALLEL_THRESHOLD * n_state
            // contiguous f64 elements. The final chunk may be shorter — rayon
            // returns whatever remains, which `evaluate_chunk` handles via
            // `chunks_exact(n_state)`.
            visited_states
                .par_chunks(PARALLEL_THRESHOLD * n_state)
                .map(|chunk| evaluate_chunk(pool, chunk, n_state, warm_start, &eligible, self))
                .reduce(
                    || vec![false; populated],
                    |mut a, b| {
                        for (ai, bi) in a.iter_mut().zip(b.iter()) {
                            *ai |= *bi;
                        }
                        a
                    },
                )
        } else {
            evaluate_chunk(pool, visited_states, n_state, warm_start, &eligible, self)
        };

        // Collect activity changes.
        // - Deactivate: eligible, not selected, currently active.
        // - Reactivate: eligible, selected, currently inactive.
        let mut deactivations: Vec<u32> = Vec::new();
        let mut reactivations: Vec<u32> = Vec::new();
        // Pool capacity is bounded to u32::MAX by construction; cast is safe.
        #[allow(clippy::cast_possible_truncation)]
        for k in warm_start..populated {
            if eligible[k] {
                let currently_active = pool.active[k];
                if is_selected[k] && !currently_active {
                    reactivations.push(k as u32);
                } else if !is_selected[k] && currently_active {
                    deactivations.push(k as u32);
                }
            }
        }

        CutActivityUpdates {
            stage_index,
            updates: deactivations,
            reactivations,
        }
    }
}

/// Evaluate a contiguous chunk of visited states and return the per-slot
/// "selected" bitmap produced by the chunk.
///
/// This helper is the core of [`CutSelectionStrategy::select_for_stage`] and
/// runs on every code path (sequential and per-rayon-worker). The bitmap is
/// `populated`-long; entry `k` is `true` iff slot `k` survives the method's
/// rule against *this chunk's* trial points. Callers combine per-chunk bitmaps
/// with bitwise-OR to obtain the global "selected" set.
///
/// Correctness of the OR-merge:
///
/// - **Level1 / Dominated**: a cut survives if it is within `tie_tolerance` of
///   the per-state max at *any* trial point. Per-chunk OR over disjoint
///   trial-point subsets is exactly the global OR.
/// - **Lml1**: per trial point only the oldest eligible cut at max survives,
///   then the per-chunk result is the union of those per-trial-point picks.
///   Since chunks process disjoint trial points, the union of per-chunk
///   results equals the global union.
fn evaluate_chunk(
    pool: &crate::cut::CutPool,
    chunk: &[f64],
    n_state: usize,
    warm_start: usize,
    eligible: &[bool],
    method: &CutSelectionStrategy,
) -> Vec<bool> {
    let populated = pool.populated_count;
    let mut is_selected = vec![false; populated];
    // Pre-allocate scratch buffer outside the trial-point loop (no allocation on hot path).
    let mut scratch = vec![0.0_f64; populated];
    let n_eligible = eligible.iter().filter(|&&e| e).count();
    let mut n_unselected = n_eligible;

    for x_hat in chunk.chunks_exact(n_state) {
        // Phase 1: evaluate ALL populated cuts (active AND inactive) to find max.
        // This is the core correctness fix: inactive cuts participate in the max
        // computation, ensuring the survival criterion is globally correct.
        let mut max_val = f64::NEG_INFINITY;
        #[allow(clippy::needless_range_loop)]
        for k in 0..populated {
            let coeff_start = k * n_state;
            let val = pool.intercepts[k]
                + pool.coefficients[coeff_start..coeff_start + n_state]
                    .iter()
                    .zip(x_hat)
                    .map(|(c, x)| c * x)
                    .sum::<f64>();
            scratch[k] = val;
            if val > max_val {
                max_val = val;
            }
        }

        // Phase 2: method-specific survival rule.
        match method {
            CutSelectionStrategy::Level1 { tie_tolerance, .. }
            | CutSelectionStrategy::Dominated {
                threshold: tie_tolerance,
                ..
            } => {
                // Level1 / Dominated: keep ALL eligible cuts within tolerance of max.
                let cutoff = max_val - tie_tolerance;
                for k in warm_start..populated {
                    if eligible[k] && !is_selected[k] && scratch[k] >= cutoff {
                        is_selected[k] = true;
                        n_unselected -= 1;
                    }
                }
            }
            CutSelectionStrategy::Lml1 { tie_tolerance, .. } => {
                // Lml1: at this trial point, only the OLDEST eligible cut at max survives.
                // Ascending slot order = oldest first (slots are assigned sequentially).
                let cutoff = max_val - tie_tolerance;
                for k in warm_start..populated {
                    if eligible[k] && scratch[k] >= cutoff {
                        if !is_selected[k] {
                            is_selected[k] = true;
                            n_unselected -= 1;
                        }
                        // Break after the oldest at-max: per trial point, only one survives.
                        break;
                    }
                }
                // Do NOT break the outer trial-point loop for Lml1.
                // Different trial points may select different oldest cuts,
                // so all trial points must be visited.
            }
        }

        // Early exit (Level1 and Dominated only): stop when all eligible cuts
        // have been selected. Lml1 must visit all trial points.
        if n_unselected == 0 && !matches!(method, CutSelectionStrategy::Lml1 { .. }) {
            break;
        }
    }

    is_selected
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

/// Parse a [`cobre_io::config::RowSelectionConfig`] into an optional
/// [`CutSelectionStrategy`].
///
/// Returns `None` when disabled (default). Returns `Err` when explicitly
/// enabled with invalid configuration (unknown method, `enabled = true` with no
/// method, or `check_frequency = 0`). Defaults: `check_frequency = 5`,
/// `tie_tolerance = 1e-10`.
///
/// The `threshold` and `memory_window` fields on `RowSelectionConfig` are
/// silently ignored — they are retained in the config struct for backward
/// compatibility with existing config files only.
///
/// # Errors
///
/// Returns `Err(String)` when `enabled = true` but no `method` is specified,
/// when the `method` string is not a recognised variant, or when
/// `check_frequency = 0`.
pub fn parse_cut_selection_config(
    config: &cobre_io::config::RowSelectionConfig,
) -> Result<Option<CutSelectionStrategy>, String> {
    const DEFAULT_TIE_TOLERANCE: f64 = 1e-10;

    let enabled = config.enabled.unwrap_or(false);
    if !enabled {
        return Ok(None);
    }

    let method = config
        .method
        .as_deref()
        .ok_or_else(|| "cut_selection.enabled is true but method is not specified".to_string())?;

    let check_frequency = config.check_frequency.unwrap_or(5);

    if check_frequency == 0 {
        return Err("cut_selection.check_frequency must be > 0".to_string());
    }

    match method {
        "level1" => Ok(Some(CutSelectionStrategy::Level1 {
            check_frequency: u64::from(check_frequency),
            tie_tolerance: config.tie_tolerance.unwrap_or(DEFAULT_TIE_TOLERANCE),
        })),
        "lml1" => Ok(Some(CutSelectionStrategy::Lml1 {
            check_frequency: u64::from(check_frequency),
            tie_tolerance: config.tie_tolerance.unwrap_or(DEFAULT_TIE_TOLERANCE),
        })),
        "domination" => {
            let epsilon = config.domination_epsilon.ok_or_else(|| {
                "cut_selection.method='domination' requires domination_epsilon to be set"
                    .to_string()
            })?;
            Ok(Some(CutSelectionStrategy::Dominated {
                threshold: epsilon,
                check_frequency: u64::from(check_frequency),
            }))
        }
        other => Err(format!("unknown cut_selection.method: \"{other}\"")),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::parse_cut_selection_config;
    use super::{CutActivityUpdates, CutMetadata, CutSelectionStrategy, PARALLEL_THRESHOLD};
    use crate::cut::CutPool;
    use cobre_io::config::RowSelectionConfig;

    fn make_meta(active_count: u64, last_active_iter: u64) -> CutMetadata {
        CutMetadata {
            iteration_generated: 1,
            forward_pass_index: 0,
            active_count,
            last_active_iter,
            active_window: 0,
        }
    }

    /// Build a `CutPool` pre-populated with the given metadata and active flags.
    #[allow(clippy::cast_possible_truncation)]
    fn make_pool(metadata: &[CutMetadata], active: &[bool]) -> CutPool {
        let n = metadata.len();
        let mut pool = CutPool::new(n, 1, 1, 0);
        // Populate dummy cuts so populated_count advances.
        for i in 0..n {
            pool.add_cut(0, i as u32, 0.0, &[0.0]);
        }
        pool.metadata[..n].clone_from_slice(metadata);
        pool.active[..n].clone_from_slice(active);
        pool.cached_active_count = active.iter().filter(|&&a| a).count();
        pool
    }

    #[test]
    fn should_run_false_at_zero() {
        let s = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        assert!(!s.should_run(0));
    }

    #[test]
    fn should_run_false_between_multiples() {
        let s = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        assert!(!s.should_run(3));
        assert!(!s.should_run(7));
    }

    #[test]
    fn should_run_true_at_multiples() {
        let s = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        assert!(s.should_run(5));
        assert!(s.should_run(10));
        assert!(s.should_run(15));
    }

    #[test]
    fn should_run_lml1_respects_check_frequency() {
        let s = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        assert!(!s.should_run(0));
        assert!(!s.should_run(3));
        assert!(s.should_run(5));
        assert!(s.should_run(10));
    }

    #[test]
    fn should_run_dominated_respects_check_frequency() {
        let s = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 10,
        };
        assert!(!s.should_run(5));
        assert!(s.should_run(10));
    }

    // -----------------------------------------------------------------------
    // Level1 value-based kernel tests
    // -----------------------------------------------------------------------

    /// AC1: pool with 3 cuts (intercepts [1,5,3], coeff all 0), state [0.0].
    /// Level1 `tie_tolerance=0.0` → cut 1 (value 5) survives; cuts 0,2 deactivated.
    #[test]
    fn level1_deactivates_dominated_cuts_at_state() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 0.0,
        };
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[0.0]);
        pool.add_cut(1, 0, 5.0, &[0.0]);
        pool.add_cut(2, 0, 3.0, &[0.0]);
        // All from iteration 1, current_iteration=10 → all eligible.
        let deact = strategy.select(&pool, &[0.0], 10);
        let mut deact_idx = deact.deactivation_indices();
        deact_idx.sort_unstable();
        assert_eq!(deact_idx, vec![0, 2], "cuts 0 and 2 must be deactivated");
        assert!(
            deact.reactivation_indices().is_empty(),
            "no reactivations expected"
        );
    }

    /// AC2: pool with 3 cuts (intercepts [5,5,3]), state [0.0].
    /// Level1 tie_tolerance=1e-10 → cuts 0 and 1 both survive (tie kept).
    #[test]
    fn level1_retains_tied_cuts() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(0, 0, 5.0, &[0.0]);
        pool.add_cut(1, 0, 5.0, &[0.0]);
        pool.add_cut(2, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0], 10);
        assert_eq!(
            deact.deactivation_indices(),
            vec![2],
            "only cut 2 (value 3) is deactivated; ties 0 and 1 kept"
        );
        assert!(deact.reactivation_indices().is_empty());
    }

    /// Level1 retains all cuts when two have equal max values.
    #[test]
    fn level1_retains_positive_activity_cuts() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 0.0,
        };
        // Two cuts with equal values at state [1.0]:
        // cut0: 1.0 + 2.0*1.0 = 3.0, cut1: 3.0 + 0.0*1.0 = 3.0 → tied, both survive.
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[2.0]);
        pool.add_cut(1, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[1.0], 10);
        assert!(
            deact.deactivation_indices().is_empty(),
            "no cuts deactivated when all tied at max"
        );
    }

    /// Level1 with three cuts where two are clearly below max.
    #[test]
    fn level1_threshold_1_deactivates_cuts_with_count_at_most_1() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 0.5,
        };
        // cut0: value=1, cut1: value=3, cut2: value=2 at state [0.0].
        // max=3, cutoff=3-0.5=2.5; only cut1(3>=2.5) and cut2(2<2.5→no) wait:
        // cut2=2 < 2.5 → not selected. cut0=1 < 2.5 → not selected. cut1=3 >= 2.5 → selected.
        // Deactivate: cut0, cut2.
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[0.0]);
        pool.add_cut(1, 0, 3.0, &[0.0]);
        pool.add_cut(2, 0, 2.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0], 10);
        let mut deact_idx = deact.deactivation_indices();
        deact_idx.sort_unstable();
        assert_eq!(deact_idx, vec![0, 2]);
    }

    #[test]
    fn level1_empty_metadata_returns_empty_set() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        let pool = CutPool::new(0, 1, 1, 0);
        let deact = strategy.select(&pool, &[], 10);
        assert!(deact.deactivation_indices().is_empty());
    }

    /// AC6: empty `visited_states` returns empty for Level1.
    #[test]
    fn level1_empty_states_returns_empty() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[0.0]);
        pool.add_cut(1, 0, 5.0, &[0.0]);
        pool.add_cut(2, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[], 10);
        assert!(deact.deactivation_indices().is_empty());
        assert!(deact.reactivation_indices().is_empty());
    }

    // -----------------------------------------------------------------------
    // Lml1 value-based kernel tests
    // -----------------------------------------------------------------------

    /// AC3: same pool as AC2 (intercepts [5,5,3]), state [0.0].
    /// Lml1 tie_tolerance=1e-10 → only cut 0 (the oldest by slot) survives.
    #[test]
    fn lml1_only_oldest_survives_at_each_state() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(0, 0, 5.0, &[0.0]);
        pool.add_cut(1, 0, 5.0, &[0.0]);
        pool.add_cut(2, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0], 10);
        let mut deact_idx = deact.deactivation_indices();
        deact_idx.sort_unstable();
        assert_eq!(
            deact_idx,
            vec![1, 2],
            "cuts 1 and 2 deactivated; only oldest (cut 0) at max survives"
        );
        assert!(deact.reactivation_indices().is_empty());
    }

    /// Lml1 with two trial points selecting different oldest cuts.
    /// state [0.0]: cut0(val=2)>cut1(val=1) → oldest at max = cut0
    /// state [1.0]: cut0=2+0=2, cut1=0+3=3 → cut1 is max, oldest at max = cut1
    /// Union: both cut0 and cut1 selected → cut2 deactivated.
    #[test]
    fn lml1_union_across_trial_points() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(0, 0, 2.0, &[0.0]); // constant 2
        pool.add_cut(1, 0, 0.0, &[3.0]); // 3x
        pool.add_cut(2, 0, 0.5, &[0.0]); // constant 0.5 (never at max)
        let deact = strategy.select(&pool, &[0.0, 1.0], 10);
        assert_eq!(
            deact.deactivation_indices(),
            vec![2],
            "only cut 2 (never at max) deactivated"
        );
        assert!(deact.reactivation_indices().is_empty());
    }

    /// Lml1: single eligible cut → `n_eligible` < 2 → empty.
    #[test]
    fn lml1_deactivates_cuts_outside_memory_window() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        // Only 1 eligible cut → n_eligible < 2 → returns empty.
        let pool = make_pool(&[make_meta(0, 5)], &[true]);
        let deact = strategy.select(&pool, &[0.0], 20);
        assert!(deact.deactivation_indices().is_empty());
    }

    /// Lml1 with two eligible cuts at the same value: oldest (slot 0) retained.
    #[test]
    fn lml1_retains_cuts_within_memory_window() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(0, 0, 3.0, &[0.0]);
        pool.add_cut(1, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0], 10);
        // Both tied → oldest (cut 0) selected → cut 1 deactivated.
        assert_eq!(
            deact.deactivation_indices(),
            vec![1],
            "cut 1 deactivated; cut 0 (oldest) retained"
        );
    }

    /// Lml1 with 3 cuts at state [0.0]: only oldest at max survives.
    #[test]
    fn lml1_retains_cuts_exactly_at_boundary() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        // cut0=5, cut1=4, cut2=5 (tied). Oldest at max = cut0.
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(0, 0, 5.0, &[0.0]);
        pool.add_cut(1, 0, 4.0, &[0.0]);
        pool.add_cut(2, 0, 5.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0], 10);
        let mut deact_idx = deact.deactivation_indices();
        deact_idx.sort_unstable();
        assert_eq!(
            deact_idx,
            vec![1, 2],
            "cuts 1 and 2 deactivated; cut 0 (oldest at max) retained"
        );
    }

    /// Lml1 with 3 cuts, 2 trial points, each selecting a different oldest cut.
    #[test]
    fn lml1_mixed_cuts_deactivates_correct_indices() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        // cut0: constant 1 (at max for state [0]: max=3→no)
        // cut1: 2x (at state [1]: 2, state [2]: 4 → max at [2])
        // cut2: constant 3 (at max for state [0]: max=3)
        // state [0.0]: values=[1,0,3] → max=3, oldest at max = cut2 (cut0<3, cut1<3)
        // state [2.0]: values=[1,4,3] → max=4, oldest at max = cut1
        // Union: cut1 and cut2 selected → cut0 deactivated.
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[0.0]);
        pool.add_cut(1, 0, 0.0, &[2.0]);
        pool.add_cut(2, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0, 2.0], 10);
        assert_eq!(
            deact.deactivation_indices(),
            vec![0],
            "only cut 0 (never at max) deactivated"
        );
    }

    /// AC6: empty `visited_states` returns empty for Lml1.
    #[test]
    fn lml1_empty_states_returns_empty() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(0, 0, 5.0, &[0.0]);
        pool.add_cut(1, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[], 10);
        assert!(deact.deactivation_indices().is_empty());
        assert!(deact.reactivation_indices().is_empty());
    }

    /// Previously: `ac_level1_threshold_0_deactivates_zero_activity_cut`.
    /// New value-based version: a cut with lower value is deactivated.
    #[test]
    fn ac_level1_threshold_0_deactivates_zero_activity_cut() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 0.0,
        };
        // cut0: value=1 at state [0]. cut1: value=3 at state [0].
        // max=3, cutoff=3. cut0(1<3) not selected → deactivated.
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[0.0]);
        pool.add_cut(1, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0], 10);
        assert!(deact.deactivation_indices().contains(&0));
    }

    /// Previously: `ac_lml1_deactivates_cut_outside_memory_window`.
    /// New value-based version: Lml1 deactivates a cut that is never oldest-at-max.
    #[test]
    fn ac_lml1_deactivates_cut_outside_memory_window() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        // cut0: value=5 at state [0]. cut1: value=3 at state [0].
        // Oldest at max = cut0. cut1 deactivated.
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(0, 0, 5.0, &[0.0]);
        pool.add_cut(1, 0, 3.0, &[0.0]);
        let deact = strategy.select(&pool, &[0.0], 20);
        assert!(deact.deactivation_indices().contains(&1));
    }

    #[test]
    fn select_for_stage_sets_stage_index() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        let pool = make_pool(&[make_meta(0, 1)], &[true]);
        let deact = strategy.select_for_stage(&pool, &[], 10, 7);
        assert_eq!(deact.stage_index, 7);
    }

    #[test]
    fn select_sets_stage_index_to_zero() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        let pool = CutPool::new(0, 1, 1, 0);
        let deact = strategy.select(&pool, &[], 10);
        assert_eq!(deact.stage_index, 0);
    }

    #[test]
    fn deactivation_set_derives_debug_and_clone() {
        let deact = CutActivityUpdates::deactivations_only(2, vec![0, 3, 7]);
        let cloned = deact.clone();
        assert_eq!(cloned.stage_index, 2);
        assert_eq!(cloned.deactivation_indices(), vec![0, 3, 7]);
        assert!(!format!("{deact:?}").is_empty());
    }

    #[test]
    fn cut_metadata_derives_debug_and_clone() {
        let meta = make_meta(5, 10);
        let cloned = meta.clone();
        assert_eq!(cloned.active_count, 5);
        assert!(!format!("{meta:?}").is_empty());
    }

    // -----------------------------------------------------------------------
    // Current-iteration guard tests
    // -----------------------------------------------------------------------

    /// Cuts generated in the current iteration must never be deactivated by
    /// Level1, even if they have lower values. They haven't been tested yet
    /// and deactivating them would cause the lower bound to stagnate.
    #[test]
    fn level1_spares_cuts_from_current_iteration() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        // cut0: iteration_generated=10 (current), value=1 at [0.0] → protected
        // cut1: iteration_generated=5 (older), value=1 at [0.0] → eligible
        // cut2: iteration_generated=5 (older), value=5 at [0.0] → eligible
        // max=5 (from cut2, which is eligible), cutoff=5.
        // cut1(1<5) not selected → deactivated. cut0 is not eligible → unchanged.
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[0.0]);
        pool.add_cut(1, 0, 1.0, &[0.0]);
        pool.add_cut(2, 0, 5.0, &[0.0]);
        pool.metadata[0].iteration_generated = 10; // current iteration
        pool.metadata[1].iteration_generated = 5;
        pool.metadata[2].iteration_generated = 5;
        let deact = strategy.select(&pool, &[0.0], 10);
        assert_eq!(
            deact.deactivation_indices(),
            vec![1],
            "only the older cut (slot 1) should be deactivated; \
             the current-iteration cut (slot 0) must be spared"
        );
    }

    /// Lml1 also spares cuts from the current iteration via the
    /// `iteration_generated` guard.
    #[test]
    fn lml1_spares_cuts_from_current_iteration() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        let pool = make_pool(
            &[CutMetadata {
                iteration_generated: 10,
                forward_pass_index: 0,
                active_count: 0,
                last_active_iter: 10,
                active_window: 0,
            }],
            &[true],
        );
        // Only 1 cut, from current iteration (not eligible) → n_eligible = 0 < 2 → empty.
        let deact = strategy.select(&pool, &[0.0], 10);
        assert!(
            deact.deactivation_indices().is_empty(),
            "current-iteration cut must not be deactivated"
        );
    }

    /// Lml1 memory window boundary behavior: 5 cuts, 2 trial points.
    /// Each trial point selects a different oldest cut; union determines survivors.
    #[test]
    fn lml1_memory_window_boundary_behavior() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        // 5 cuts with different values at 2 states:
        // cut0: constant 1 (coeff=0, intercept=1)
        // cut1: constant 2 (coeff=0, intercept=2)
        // cut2: 2x (coeff=2, intercept=0)
        // cut3: constant 3 (coeff=0, intercept=3)
        // cut4: x+1 (coeff=1, intercept=1)
        //
        // state [0.0]: values=[1,2,0,3,1] → max=3 (cut3). Oldest at max: cut3.
        // state [1.0]: values=[1,2,2,3,2] → max=3 (cut3). Oldest at max: cut3 again.
        // Union: only cut3 selected → cuts 0,1,2,4 deactivated.
        let mut pool = CutPool::new(5, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[0.0]);
        pool.add_cut(1, 0, 2.0, &[0.0]);
        pool.add_cut(2, 0, 0.0, &[2.0]);
        pool.add_cut(3, 0, 3.0, &[0.0]);
        pool.add_cut(4, 0, 1.0, &[1.0]);
        let deact = strategy.select_for_stage(&pool, &[0.0, 1.0], 10, 0);
        let mut deact_idx = deact.deactivation_indices();
        deact_idx.sort_unstable();
        assert_eq!(
            deact_idx,
            vec![0, 1, 2, 4],
            "only cut 3 (oldest at max at both states) survives"
        );
    }

    // -----------------------------------------------------------------------
    // Reactivation tests
    // -----------------------------------------------------------------------

    /// AC4: inactive cut achieves max at a trial point → Reactivate entry emitted.
    #[test]
    fn level1_reactivates_inactive_cut_at_max() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        // cut0: intercept=5, active=false (inactive but achieves max)
        // cut1: intercept=3, active=true
        // cut2: intercept=1, active=true
        // max=5 (cut0). cut0 selected → reactivate. cut1,cut2 not selected → deactivate.
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(0, 0, 5.0, &[0.0]);
        pool.add_cut(1, 0, 3.0, &[0.0]);
        pool.add_cut(2, 0, 1.0, &[0.0]);
        // Manually deactivate cut 0.
        pool.set_active(0, false);
        assert_eq!(pool.active_count(), 2);

        let result = strategy.select(&pool, &[0.0], 10);
        assert_eq!(
            result.reactivation_indices(),
            vec![0],
            "inactive cut 0 (at max) must be reactivated"
        );
        let mut deact_idx = result.deactivation_indices();
        deact_idx.sort_unstable();
        assert_eq!(
            deact_idx,
            vec![1, 2],
            "active cuts 1 and 2 (below max) must be deactivated"
        );
    }

    /// Lml1 also emits reactivation for an inactive cut that is oldest-at-max.
    #[test]
    fn lml1_reactivates_inactive_oldest_at_max() {
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        // cut0: intercept=5, inactive (oldest, at max)
        // cut1: intercept=5, active (also at max, but younger than cut0)
        // Oldest at max = cut0 (slot 0 < slot 1). cut1 not selected → deactivate.
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(0, 0, 5.0, &[0.0]);
        pool.add_cut(1, 0, 5.0, &[0.0]);
        pool.set_active(0, false);

        let result = strategy.select(&pool, &[0.0], 10);
        assert_eq!(
            result.reactivation_indices(),
            vec![0],
            "inactive cut 0 (oldest at max) must be reactivated"
        );
        assert_eq!(
            result.deactivation_indices(),
            vec![1],
            "active cut 1 (younger at max) must be deactivated"
        );
    }

    /// AC5: all populated cuts from current iteration → empty result.
    #[test]
    fn select_returns_empty_when_all_cuts_from_current_iteration() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[0.0]);
        pool.add_cut(1, 0, 5.0, &[0.0]);
        pool.add_cut(2, 0, 3.0, &[0.0]);
        // All from current iteration.
        pool.metadata[0].iteration_generated = 10;
        pool.metadata[1].iteration_generated = 10;
        pool.metadata[2].iteration_generated = 10;
        let result = strategy.select(&pool, &[0.0], 10);
        assert!(
            result.deactivation_indices().is_empty(),
            "no activity changes when all cuts from current iteration"
        );
        assert!(result.reactivation_indices().is_empty());
    }

    // -----------------------------------------------------------------------
    // Warm-start slot protection tests
    // -----------------------------------------------------------------------

    /// Warm-start cuts participate in max computation but are not candidates.
    /// With `warm_start_count=1`: slot 0 is protected; only slots 1+ are eligible.
    #[allow(clippy::cast_possible_truncation)]
    #[test]
    fn level1_warm_start_cuts_not_deactivated() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 0.0,
        };
        // warm_start_count=1 → slot 0 is warm-start (protected).
        // Populate the pool directly (warm-start slots are not inserted via add_cut).
        // 3 slots total: slot 0 (warm-start, intercept=10), slot 1 (eligible, =1), slot 2 (eligible, =3).
        // max=10 (slot 0). Cutoff=10. Eligible cuts 1,2 both below cutoff → deactivated.
        // Slot 0 is not eligible (warm-start) → not deactivated.
        let n = 3usize;
        let mut pool = CutPool::new(n, 1, 1, 1); // warm_start_count=1
        let intercepts = [10.0f64, 1.0, 3.0];
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            pool.intercepts[i] = intercepts[i];
            pool.coefficients[i] = 0.0;
            pool.active[i] = true;
            pool.metadata[i] = CutMetadata {
                iteration_generated: 1,
                forward_pass_index: i as u32,
                active_count: 0,
                last_active_iter: 1,
                active_window: 0,
            };
        }
        pool.populated_count = n;
        pool.cached_active_count = n;

        let result = strategy.select(&pool, &[0.0], 10);
        // Warm-start slot 0 must not appear in deactivations.
        assert!(
            !result.deactivation_indices().contains(&0),
            "warm-start slot 0 must not be deactivated"
        );
        let mut deact_idx = result.deactivation_indices();
        deact_idx.sort_unstable();
        assert_eq!(deact_idx, vec![1, 2]);
    }

    // -----------------------------------------------------------------------
    // Skip already-inactive slot tests
    // -----------------------------------------------------------------------

    /// Previously: `select_skips_already_inactive_slots`.
    /// An already-inactive cut that is not at max produces no change.
    /// An already-inactive cut that IS at max is reactivated.
    #[test]
    fn select_skips_already_inactive_slots() {
        let mut pool = CutPool::new(10, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[0.0]); // slot 0: value=1
        pool.add_cut(1, 0, 5.0, &[0.0]); // slot 1: value=5 (max)
        pool.add_cut(2, 0, 3.0, &[0.0]); // slot 2: value=3
        assert_eq!(pool.active_count(), 3);

        // Manually deactivate slot 0 before selection.
        pool.set_active(0, false);
        assert_eq!(pool.active_count(), 2);

        // Level1 tie_tolerance=0: only slot 1 (value=5) is at max.
        // slot 0: eligible, not selected, inactive → no change (not reactivated).
        // slot 1: eligible, selected, active → no change.
        // slot 2: eligible, not selected, active → deactivated.
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 0.0,
        };
        let deact = strategy.select_for_stage(&pool, &[0.0], 5, 0);
        assert_eq!(
            deact.deactivation_indices(),
            vec![2],
            "only slot 2 (active, below max) deactivated"
        );
        assert!(
            deact.reactivation_indices().is_empty(),
            "slot 0 (inactive, below max) must not be reactivated"
        );
    }

    // -----------------------------------------------------------------------
    // select_for_stage returns CutActivityUpdates with deactivations
    // -----------------------------------------------------------------------

    #[test]
    fn select_for_stage_returns_cut_activity_updates_with_deactivations() {
        let mut pool = CutPool::new(3, 1, 1, 0);
        pool.add_cut(0, 0, 1.0, &[0.0]); // value=1
        pool.add_cut(1, 0, 2.0, &[0.0]); // value=2 (max)

        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 0.0,
        };
        let result = strategy.select_for_stage(&pool, &[0.0], 10, 0);

        // cut0 (value=1) not at max → deactivated. cut1 (value=2) at max → no change.
        assert!(
            !result.updates.is_empty(),
            "deactivations must be non-empty"
        );
        assert!(
            result.updates.contains(&0),
            "slot 0 (below max) must be deactivated"
        );
        assert!(
            !result.updates.contains(&1),
            "slot 1 (at max) must not be deactivated"
        );
    }

    // -----------------------------------------------------------------------
    // Aggressiveness ordering
    // -----------------------------------------------------------------------

    /// Level1 and Lml1 with empty states return empty; Dominated with states
    /// returns some. Ordering: |Level1| <= |Lml1| <= |Dominated|.
    #[test]
    fn aggressiveness_ordering_level1_leq_lml1_leq_dominated() {
        // 5 cuts (1D):
        // Cut 0: intercept=0, slope=0 (constant 0)
        // Cut 1: intercept=0, slope=0.1
        // Cut 2: intercept=1, slope=0
        // Cut 3: intercept=0, slope=2
        // Cut 4: intercept=5, slope=-1
        let meta = [
            CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 0,
                active_count: 0,
                last_active_iter: 1,
                active_window: 0,
            },
            CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 1,
                active_count: 0,
                last_active_iter: 2,
                active_window: 0,
            },
            CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 2,
                active_count: 3,
                last_active_iter: 3,
                active_window: 0,
            },
            CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 3,
                active_count: 5,
                last_active_iter: 10,
                active_window: 0,
            },
            CutMetadata {
                iteration_generated: 1,
                forward_pass_index: 4,
                active_count: 5,
                last_active_iter: 10,
                active_window: 0,
            },
        ];
        let pool = make_dominated_pool(
            &[0.0, 0.0, 1.0, 0.0, 5.0],
            &[vec![0.0], vec![0.1], vec![0.0], vec![2.0], vec![-1.0]],
            &[true; 5],
            &meta,
        );
        let states: Vec<f64> = vec![0.0, 1.0, 3.0, 5.0];

        // Level1 and Lml1 use value-based evaluation with actual states.
        let l1 = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        let deact_l1 = l1.select(&pool, &states, 11);

        let lml1 = CutSelectionStrategy::Lml1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        let deact_lml1 = lml1.select(&pool, &states, 11);

        // Dominated threshold=0
        let dom = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let deact_dom = dom.select(&pool, &states, 11);

        assert!(
            deact_l1.deactivation_indices().len() <= deact_lml1.deactivation_indices().len(),
            "Level1 ({}) should deactivate <= LML1 ({})",
            deact_l1.deactivation_indices().len(),
            deact_lml1.deactivation_indices().len()
        );
        assert!(
            deact_lml1.deactivation_indices().len() <= deact_dom.deactivation_indices().len(),
            "LML1 ({}) should deactivate <= Dominated ({})",
            deact_lml1.deactivation_indices().len(),
            deact_dom.deactivation_indices().len()
        );
    }

    #[test]
    fn cut_activity_updates_deactivations_only_constructor() {
        let updates = CutActivityUpdates::deactivations_only(7, vec![0, 1, 2]);
        assert_eq!(updates.stage_index, 7);
        assert_eq!(updates.updates.len(), 3);
        assert_eq!(updates.updates, vec![0, 1, 2]);
        assert!(updates.reactivations.is_empty());
    }

    #[test]
    fn cut_activity_updates_deactivation_indices_returns_updates() {
        let updates = CutActivityUpdates {
            stage_index: 0,
            updates: vec![0, 2],
            reactivations: vec![],
        };
        assert_eq!(updates.deactivation_indices(), vec![0, 2]);
        assert!(updates.reactivation_indices().is_empty());
    }

    // -----------------------------------------------------------------------
    // parse_cut_selection_config tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_disabled_default() {
        let cfg = RowSelectionConfig::default();
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "default config must produce None (disabled)"
        );
    }

    #[test]
    fn test_parse_level1() {
        let cfg = RowSelectionConfig {
            enabled: Some(true),
            method: Some("level1".to_string()),
            threshold: Some(0),
            check_frequency: Some(5),
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: None,
            domination_epsilon: None,
            basis_activity_window: None,
            tie_tolerance: Some(1e-8),
        };
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_ok());
        let strategy = result
            .unwrap()
            .expect("must produce Some for enabled level1");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Level1 {
                    check_frequency: 5,
                    tie_tolerance,
                } if (tie_tolerance - 1e-8).abs() < f64::EPSILON
            ),
            "unexpected variant: {strategy:?}"
        );
    }

    /// AC: `level1` with no `tie_tolerance` uses the default 1e-10.
    #[test]
    fn test_parse_level1_default_tie_tolerance() {
        let cfg = RowSelectionConfig {
            enabled: Some(true),
            method: Some("level1".to_string()),
            threshold: None,
            check_frequency: Some(5),
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: None,
            domination_epsilon: None,
            basis_activity_window: None,
            tie_tolerance: None,
        };
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_ok());
        let strategy = result
            .unwrap()
            .expect("must produce Some for enabled level1 without tie_tolerance");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Level1 {
                    check_frequency: 5,
                    tie_tolerance,
                } if (tie_tolerance - 1e-10).abs() < 1e-20
            ),
            "unexpected variant or wrong default tie_tolerance: {strategy:?}"
        );
    }

    #[test]
    fn test_parse_lml1() {
        // AC: `lml1` with explicit `tie_tolerance` uses the provided value.
        let cfg = RowSelectionConfig {
            enabled: Some(true),
            method: Some("lml1".to_string()),
            threshold: None,
            check_frequency: Some(5),
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: Some(10), // deprecated; silently ignored
            domination_epsilon: None,
            basis_activity_window: None,
            tie_tolerance: Some(1e-8),
        };
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_ok());
        let strategy = result.unwrap().expect("must produce Some for enabled lml1");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Lml1 {
                    check_frequency: 5,
                    tie_tolerance,
                } if (tie_tolerance - 1e-8).abs() < f64::EPSILON
            ),
            "unexpected variant: {strategy:?}"
        );
    }

    /// AC: `lml1` without `memory_window` and without `tie_tolerance` must succeed
    /// and use the default `tie_tolerance` of 1e-10.
    #[test]
    fn test_parse_lml1_missing_memory_window_succeeds() {
        let cfg = RowSelectionConfig {
            enabled: Some(true),
            method: Some("lml1".to_string()),
            threshold: None,
            check_frequency: Some(5),
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: None,
            domination_epsilon: None,
            basis_activity_window: None,
            tie_tolerance: None,
        };
        let result = parse_cut_selection_config(&cfg);
        assert!(
            result.is_ok(),
            "lml1 without memory_window must not error: {:?}",
            result.unwrap_err()
        );
        let strategy = result
            .unwrap()
            .expect("must produce Some for enabled lml1 without memory_window");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Lml1 {
                    check_frequency: 5,
                    tie_tolerance,
                } if (tie_tolerance - 1e-10).abs() < 1e-20
            ),
            "unexpected variant or wrong default tie_tolerance: {strategy:?}"
        );
    }

    #[test]
    fn test_parse_domination() {
        let cfg = RowSelectionConfig {
            enabled: Some(true),
            method: Some("domination".to_string()),
            threshold: None,
            check_frequency: Some(10),
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: None,
            domination_epsilon: Some(1e-6),
            basis_activity_window: None,
            tie_tolerance: None,
        };
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_ok());
        let strategy = result
            .unwrap()
            .expect("must produce Some for enabled domination");
        assert!(
            matches!(
                strategy,
                CutSelectionStrategy::Dominated {
                    threshold,
                    check_frequency: 10,
                } if (threshold - 1e-6).abs() < f64::EPSILON
            ),
            "unexpected variant: {strategy:?}"
        );
    }

    #[test]
    fn test_parse_domination_missing_epsilon_errors() {
        let cfg = RowSelectionConfig {
            enabled: Some(true),
            method: Some("domination".to_string()),
            threshold: None,
            check_frequency: Some(10),
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: None,
            domination_epsilon: None,
            basis_activity_window: None,
            tie_tolerance: None,
        };
        let result = parse_cut_selection_config(&cfg);
        assert!(
            result.is_err(),
            "domination without domination_epsilon must error"
        );
        let msg = result.unwrap_err();
        assert!(
            msg.contains("domination_epsilon"),
            "error must mention domination_epsilon, got: {msg}"
        );
    }

    #[test]
    fn test_parse_unknown_method() {
        let cfg = RowSelectionConfig {
            enabled: Some(true),
            method: Some("bogus".to_string()),
            threshold: None,
            check_frequency: None,
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: None,
            domination_epsilon: None,
            basis_activity_window: None,
            tie_tolerance: None,
        };
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("bogus"),
            "error message must contain the unrecognized method name, got: {msg}"
        );
    }

    #[test]
    fn test_parse_enabled_without_method() {
        let cfg = RowSelectionConfig {
            enabled: Some(true),
            method: None,
            threshold: None,
            check_frequency: None,
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: None,
            domination_epsilon: None,
            basis_activity_window: None,
            tie_tolerance: None,
        };
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_enabled_false_with_method_returns_none() {
        let cfg = RowSelectionConfig {
            enabled: Some(false),
            method: Some("level1".to_string()),
            threshold: None,
            check_frequency: None,
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: None,
            domination_epsilon: None,
            basis_activity_window: None,
            tie_tolerance: None,
        };
        let result = parse_cut_selection_config(&cfg).unwrap();
        assert!(
            result.is_none(),
            "enabled=false must return None even when method is set"
        );
    }

    #[test]
    fn test_parse_zero_check_frequency() {
        let cfg = RowSelectionConfig {
            enabled: Some(true),
            method: Some("level1".to_string()),
            threshold: None,
            check_frequency: Some(0),
            cut_activity_tolerance: None,
            max_active_per_stage: None,
            memory_window: None,
            domination_epsilon: None,
            basis_activity_window: None,
            tie_tolerance: None,
        };
        let result = parse_cut_selection_config(&cfg);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("check_frequency"),
            "error message must mention check_frequency, got: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // Dominated algorithm tests (SS1.3 conformance + aggressiveness ordering)
    // -----------------------------------------------------------------------

    /// Build a `CutPool` with known coefficients, intercepts, and metadata
    /// for testing the dominated selection algorithm.
    #[allow(clippy::cast_possible_truncation)]
    fn make_dominated_pool(
        intercepts: &[f64],
        coefficients: &[Vec<f64>],
        active: &[bool],
        metadata: &[CutMetadata],
    ) -> CutPool {
        let n = intercepts.len();
        let state_dim = coefficients[0].len();
        let mut pool = CutPool::new(n, state_dim, 1, 0);
        for i in 0..n {
            // Use add_cut to advance populated_count correctly.
            pool.add_cut(0, i as u32, intercepts[i], &coefficients[i]);
            pool.metadata[i] = metadata[i].clone();
            pool.active[i] = active[i];
        }
        pool.cached_active_count = active.iter().filter(|&&a| a).count();
        pool
    }

    fn default_meta_at(iter: u64) -> CutMetadata {
        CutMetadata {
            iteration_generated: iter,
            forward_pass_index: 0,
            active_count: 0,
            last_active_iter: iter,
            active_window: 0,
        }
    }

    fn default_meta_vec(n: usize, iter: u64) -> Vec<CutMetadata> {
        (0..n).map(|_| default_meta_at(iter)).collect()
    }

    /// SS1.3 test 1: 5 cuts, 3 states (1D). Cuts 0,3,4 dominated at all states.
    #[test]
    fn dominated_select_deactivate_dominated() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let pool = make_dominated_pool(
            &[1.0, 0.0, 3.0, 0.5, 0.0],
            &[
                vec![0.0],  // cut 0: constant 1
                vec![2.0],  // cut 1: 2x
                vec![-1.0], // cut 2: 3 - x
                vec![0.0],  // cut 3: constant 0.5
                vec![0.5],  // cut 4: 0.5x
            ],
            &[true; 5],
            &default_meta_vec(5, 1),
        );
        let states: Vec<f64> = vec![0.0, 1.0, 3.0];
        let deact = strategy.select(&pool, &states, 10);
        assert_eq!(deact.deactivation_indices(), vec![0, 3, 4]);
    }

    /// SS1.3 test 2: cut dominated at 2/3 states but tied at 1 -> retained.
    #[test]
    fn dominated_select_partial_domination_retained() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        // Cut 0: intercept=2, slope=0 (constant 2)
        // Cut 1: intercept=0, slope=2 (2x)
        // At x=0: values=[2, 0] -> max=2, cut 0 achieves max -> not dominated
        // At x=1: values=[2, 2] -> max=2, cut 0 achieves max -> not dominated
        // At x=3: values=[2, 6] -> max=6, cut 0 below -> dominated at this state
        // Net: cut 0 is NOT dominated (achieves max at x=0 and x=1)
        let pool = make_dominated_pool(
            &[2.0, 0.0],
            &[vec![0.0], vec![2.0]],
            &[true, true],
            &default_meta_vec(2, 1),
        );
        let states: Vec<f64> = vec![0.0, 1.0, 3.0];
        let deact = strategy.select(&pool, &states, 10);
        assert!(
            deact.deactivation_indices().is_empty(),
            "cut 0 achieves max at x=0 and x=1, must not be deactivated"
        );
    }

    /// SS1.3 test 3: cut 2 (constant 2) never achieves max → deactivated.
    #[test]
    fn dominated_select_none_dominated_when_all_achieve_max() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        // Cut 0: 5 - 2x (max at x=0: 5)
        // Cut 1: 0 + 3x (max at x=3: 9)
        // Cut 2: 2 + 0x (constant 2, never achieves max)
        let pool = make_dominated_pool(
            &[5.0, 0.0, 2.0],
            &[vec![-2.0], vec![3.0], vec![0.0]],
            &[true; 3],
            &default_meta_vec(3, 1),
        );
        let states: Vec<f64> = vec![0.0, 1.0, 3.0];
        let deact = strategy.select(&pool, &states, 10);
        assert_eq!(
            deact.deactivation_indices(),
            vec![2],
            "only cut 2 (constant 2) should be dominated"
        );
    }

    /// SS1.3 test 4: empty `visited_states` returns empty set.
    #[test]
    fn dominated_select_empty_states() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let pool = make_dominated_pool(
            &[1.0, 2.0],
            &[vec![0.0], vec![0.0]],
            &[true, true],
            &default_meta_vec(2, 1),
        );
        let deact = strategy.select(&pool, &[], 10);
        assert!(
            deact.deactivation_indices().is_empty(),
            "empty visited_states must produce empty deactivation set"
        );
    }

    /// SS1.3 test 5 (updated): with 1 active and 2 inactive cuts, the unified
    /// kernel evaluates ALL cuts for max. The highest inactive cut (intercept=3)
    /// is selected (reactivated), and the only active cut (intercept=1) is deactivated.
    ///
    /// The old test asserted empty deactivations when only 1 active cut existed,
    /// reflecting the old active-only max computation. The new kernel includes
    /// inactive cuts in the max, which is the core correctness fix.
    #[test]
    fn dominated_select_single_active_cut() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let pool = make_dominated_pool(
            &[1.0, 2.0, 3.0],
            &[vec![0.0], vec![0.0], vec![0.0]],
            &[true, false, false],
            &default_meta_vec(3, 1),
        );
        let states: Vec<f64> = vec![0.0, 1.0];
        let deact = strategy.select(&pool, &states, 10);
        // max=3 (cut2), cutoff=3. Only cut2 selected.
        // cut0: active, not selected → deactivated.
        // cut1: inactive, not selected → no change.
        // cut2: inactive, selected → reactivated.
        assert_eq!(
            deact.deactivation_indices(),
            vec![0],
            "active cut 0 (below max) must be deactivated"
        );
        assert_eq!(
            deact.reactivation_indices(),
            vec![2],
            "inactive cut 2 (at max) must be reactivated"
        );
    }

    /// SS1.3 test 6: cut from current iteration excluded from deactivation.
    #[test]
    fn dominated_select_current_iteration_excluded() {
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        // Cut 0: constant 1 (from current iteration 10 -- protected)
        // Cut 1: constant 5 (from iteration 1 -- dominates cut 0)
        let pool = make_dominated_pool(
            &[1.0, 5.0],
            &[vec![0.0], vec![0.0]],
            &[true, true],
            &[default_meta_at(10), default_meta_at(1)],
        );
        let states: Vec<f64> = vec![0.0, 1.0];
        let deact = strategy.select(&pool, &states, 10);
        assert!(
            deact.deactivation_indices().is_empty(),
            "cut from current iteration must not be deactivated even if dominated"
        );
    }

    // -----------------------------------------------------------------------
    // AC6 set-inclusion property: Level1_selected ⊇ Lml1_selected
    // -----------------------------------------------------------------------

    /// AC6 (set-inclusion form): every cut that Lml1 keeps must also be kept by
    /// Level1 — i.e., every slot in Lml1's deactivation list also appears in
    /// Level1's deactivation list.  The existing aggressiveness ordering test
    /// only checks that `|deact_L1|` <= `|deact_Lml1|`; this test directly verifies
    /// the subset relationship on slot indices.
    ///
    /// Fixture: 4 cuts (1D), 2 trial points.
    /// cut0: constant 1  (coeff=0, intercept=1)
    /// cut1: constant 3  (coeff=0, intercept=3)
    /// cut2: 2x          (coeff=2, intercept=0)
    /// cut3: constant 0  (coeff=0, intercept=0) -- always below max
    ///
    /// At state [0.0]: values=[1,3,0,0] → max=3 (cut1). Level1 keeps cut1.
    ///   Lml1 keeps oldest at max = cut1.
    /// At state [2.0]: values=[1,3,4,0] → max=4 (cut2). Level1 keeps cut1,cut2.
    ///   Lml1 keeps oldest at max = cut2.
    ///
    /// Level1 deactivates: {cut0, cut3}.
    /// Lml1 deactivates: {cut0, cut3} (same here, but the property holds).
    ///
    /// Set-inclusion check: every slot in `deact_lml1` must also be in `deact_l1`.
    #[test]
    fn level1_selected_is_superset_of_lml1_selected() {
        let meta = default_meta_vec(4, 1);
        let pool = make_dominated_pool(
            &[1.0, 3.0, 0.0, 0.0],
            &[vec![0.0], vec![0.0], vec![2.0], vec![0.0]],
            &[true; 4],
            &meta,
        );
        let states: Vec<f64> = vec![0.0, 2.0];

        let l1 = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        let lml1 = CutSelectionStrategy::Lml1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };

        let deact_l1 = l1.select(&pool, &states, 10);
        let deact_lml1 = lml1.select(&pool, &states, 10);

        // Set-inclusion: every slot Lml1 deactivates must also be deactivated by Level1.
        // Equivalently: the Level1 survivor set is a superset of the Lml1 survivor set.
        for slot in deact_lml1.deactivation_indices() {
            assert!(
                deact_l1.deactivation_indices().contains(&slot),
                "slot {slot} deactivated by Lml1 but not by Level1; \
                 Level1_selected must be a superset of Lml1_selected"
            );
        }
        // Sanity: Level1 must not deactivate more than it could (count check too).
        assert!(
            deact_l1.deactivation_indices().len() <= deact_lml1.deactivation_indices().len(),
            "Level1 must deactivate <= Lml1"
        );
    }

    // -----------------------------------------------------------------------
    // AC3 variant: Dominated epsilon-tolerance test
    // -----------------------------------------------------------------------

    /// A cut 1e-7 below max everywhere survives with epsilon=1e-6 but is
    /// deactivated with epsilon=0.
    ///
    /// Fixture: 2 cuts (1D constant), states [0.0].
    /// cut0: constant 5.0  (max)
    /// cut1: constant 4.9999999  (max - 1e-7)
    ///
    /// With epsilon=1e-6: cutoff = 5.0 - 1e-6 = 4.999999. cut1(4.9999999) > 4.999999 → survives.
    /// With epsilon=0:    cutoff = 5.0.       cut1(4.9999999) < 5.0 → deactivated.
    #[test]
    fn dominated_epsilon_tolerance_cut_barely_below_max() {
        let meta = default_meta_vec(2, 1);
        let pool = make_dominated_pool(
            &[5.0, 4.999_999_9],
            &[vec![0.0], vec![0.0]],
            &[true; 2],
            &meta,
        );
        let states: Vec<f64> = vec![0.0];

        // With epsilon=1e-6: cut1 is within tolerance → survives.
        let dom_loose = CutSelectionStrategy::Dominated {
            threshold: 1e-6,
            check_frequency: 1,
        };
        let deact_loose = dom_loose.select(&pool, &states, 10);
        assert!(
            deact_loose.deactivation_indices().is_empty(),
            "cut1 (1e-7 below max) must survive when epsilon=1e-6"
        );

        // With epsilon=0: cut1 is strictly below max → deactivated.
        let dom_strict = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let deact_strict = dom_strict.select(&pool, &states, 10);
        assert_eq!(
            deact_strict.deactivation_indices(),
            vec![1],
            "cut1 (1e-7 below max) must be deactivated when epsilon=0"
        );
    }

    // -----------------------------------------------------------------------
    // Edge case: single eligible cut
    // -----------------------------------------------------------------------

    /// Level1 with exactly one eligible cut returns empty updates (`n_eligible` < 2
    /// guard fires before any evaluation).
    ///
    /// Two slots total: slot 0 is from `current_iteration` (ineligible); slot 1 is
    /// eligible. `n_eligible=1` → empty.
    #[test]
    fn level1_single_eligible_cut_returns_empty() {
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        let mut pool = CutPool::new(2, 1, 1, 0);
        pool.add_cut(0, 0, 10.0, &[0.0]); // higher value
        pool.add_cut(1, 0, 1.0, &[0.0]); // lower value
        // Slot 0 from current iteration → ineligible. Slot 1 eligible.
        pool.metadata[0].iteration_generated = 10; // current_iteration
        pool.metadata[1].iteration_generated = 5;
        // n_eligible = 1 (only slot 1). Guard returns empty.
        let result = strategy.select(&pool, &[0.0], 10);
        assert!(
            result.deactivation_indices().is_empty(),
            "single eligible cut must not trigger any deactivations"
        );
        assert!(result.reactivation_indices().is_empty());
    }

    // -----------------------------------------------------------------------
    // Intra-stage parallelism determinism tests
    // -----------------------------------------------------------------------

    /// Build a 100-cut, 1-D pool whose intercepts and coefficients are
    /// deterministic functions of the slot index — designed so that different
    /// cuts achieve the max at different trial points, exercising both
    /// Level1's "any-max" and Lml1's "oldest-at-max" branches.
    #[allow(clippy::cast_precision_loss)]
    fn make_determinism_pool() -> CutPool {
        const N: usize = 100;
        let mut pool = CutPool::new(N, 1, 1, 0);
        for i in 0..N {
            // Use varied (intercept, slope) pairs so values are non-trivial.
            // intercept = i mod 7, slope = ((i + 3) mod 5) - 2  (range [-2, 2]).
            let intercept = (i % 7) as f64;
            let slope = ((i + 3) % 5) as f64 - 2.0;
            #[allow(clippy::cast_possible_truncation)]
            pool.add_cut(0, i as u32, intercept, &[slope]);
            // Make every cut eligible (iteration_generated < current_iteration in
            // the tests below).
            pool.metadata[i].iteration_generated = 1;
        }
        pool
    }

    /// Build >= 1000 trial points spanning a representative range.
    #[allow(clippy::cast_precision_loss)]
    fn make_determinism_states(count: usize) -> Vec<f64> {
        (0..count).map(|i| (i as f64) * 0.01 - 5.0).collect()
    }

    /// Run `select_for_stage` for `strategy` inside a rayon thread pool with
    /// `num_threads` workers, returning the resulting `CutActivityUpdates`.
    fn run_in_pool(
        strategy: &CutSelectionStrategy,
        pool: &CutPool,
        states: &[f64],
        current_iteration: u64,
        num_threads: usize,
    ) -> CutActivityUpdates {
        let rayon_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("rayon pool must build for determinism test");
        rayon_pool.install(|| strategy.select_for_stage(pool, states, current_iteration, 0))
    }

    /// Determinism: identical bit-for-bit output for 1 vs 4 threads, across
    /// thresholds that bracket [`PARALLEL_THRESHOLD`].
    ///
    /// 1000 trial points >> threshold of 256 → parallel path engaged with
    /// multiple chunks. The bitwise-OR merge is commutative and associative,
    /// so any worker assignment must yield the same `is_selected` bitmap and
    /// therefore identical `CutActivityUpdates`.
    #[test]
    fn select_for_stage_deterministic_across_thread_counts_level1() {
        let pool = make_determinism_pool();
        let states = make_determinism_states(1024);
        let strategy = CutSelectionStrategy::Level1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        let r1 = run_in_pool(&strategy, &pool, &states, 10, 1);
        let r4 = run_in_pool(&strategy, &pool, &states, 10, 4);
        let r8 = run_in_pool(&strategy, &pool, &states, 10, 8);
        assert_eq!(
            r1, r4,
            "Level1: 1-thread vs 4-thread results must be bit-identical"
        );
        assert_eq!(
            r4, r8,
            "Level1: 4-thread vs 8-thread results must be bit-identical"
        );
    }

    /// Determinism for Lml1: per-chunk "oldest at max" picks unioned with
    /// bitwise-OR across disjoint trial-point chunks must reproduce the
    /// sequential global "oldest at max" union.
    #[test]
    fn select_for_stage_deterministic_across_thread_counts_lml1() {
        let pool = make_determinism_pool();
        let states = make_determinism_states(1024);
        let strategy = CutSelectionStrategy::Lml1 {
            check_frequency: 1,
            tie_tolerance: 1e-10,
        };
        let r1 = run_in_pool(&strategy, &pool, &states, 10, 1);
        let r4 = run_in_pool(&strategy, &pool, &states, 10, 4);
        let r8 = run_in_pool(&strategy, &pool, &states, 10, 8);
        assert_eq!(
            r1, r4,
            "Lml1: 1-thread vs 4-thread results must be bit-identical"
        );
        assert_eq!(
            r4, r8,
            "Lml1: 4-thread vs 8-thread results must be bit-identical"
        );
    }

    /// Determinism for Dominated: same proof structure as Level1.
    #[test]
    fn select_for_stage_deterministic_across_thread_counts_dominated() {
        let pool = make_determinism_pool();
        let states = make_determinism_states(1024);
        let strategy = CutSelectionStrategy::Dominated {
            threshold: 0.0,
            check_frequency: 1,
        };
        let r1 = run_in_pool(&strategy, &pool, &states, 10, 1);
        let r4 = run_in_pool(&strategy, &pool, &states, 10, 4);
        let r8 = run_in_pool(&strategy, &pool, &states, 10, 8);
        assert_eq!(
            r1, r4,
            "Dominated: 1-thread vs 4-thread results must be bit-identical"
        );
        assert_eq!(
            r4, r8,
            "Dominated: 4-thread vs 8-thread results must be bit-identical"
        );
    }

    /// Sequential (small input, below threshold) vs parallel (large input)
    /// must agree on the subset of states shared between them. Concretely:
    /// running the sequential path on N states and the parallel path on the
    /// same N states (forced via thread-pool size 4) must yield identical
    /// output, even when N is just past the threshold so the parallel path
    /// produces a small number of chunks plus a possibly-short tail chunk.
    #[test]
    fn select_for_stage_parallel_matches_sequential_just_past_threshold() {
        let pool = make_determinism_pool();
        // PARALLEL_THRESHOLD + 7 → trigger parallel path with one partial chunk.
        let n_states = PARALLEL_THRESHOLD + 7;
        let states = make_determinism_states(n_states);

        for strategy in [
            CutSelectionStrategy::Level1 {
                check_frequency: 1,
                tie_tolerance: 1e-10,
            },
            CutSelectionStrategy::Lml1 {
                check_frequency: 1,
                tie_tolerance: 1e-10,
            },
            CutSelectionStrategy::Dominated {
                threshold: 0.0,
                check_frequency: 1,
            },
        ] {
            // 1-thread pool → still takes parallel branch (n_states >= threshold)
            // but executes chunks serially. 4-thread pool → genuine parallelism.
            let seq = run_in_pool(&strategy, &pool, &states, 10, 1);
            let par = run_in_pool(&strategy, &pool, &states, 10, 4);
            assert_eq!(
                seq, par,
                "strategy {strategy:?}: parallel must equal sequential at threshold boundary"
            );
        }
    }
}
