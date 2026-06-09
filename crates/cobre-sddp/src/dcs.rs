//! Dynamic Cut Selection (DCS) hyperparameters as a hot-path value type.
//!
//! DCS is a per-solve lazy selection loop: rather than deactivating cuts at the
//! pool level, it grows the LP's active cut set lazily within each solve until
//! no candidate cut is violated by more than `epsilon_viol` (or a bounded
//! inner-iteration cap is hit). See
//! `docs/design/cut-selection-dynamic-redesign.md` for the full design.
//!
//! [`DcsParams`] is a small, `Copy`, allocation-free carrier for the DCS
//! hyperparameters. It mirrors the values stored on
//! [`CutSelectionStrategy::Dynamic`](crate::cut_selection::CutSelectionStrategy::Dynamic)
//! (the serde-stable config representation) but is intentionally decoupled from
//! it so the hot path is not coupled to config parsing. Construct one from a
//! strategy via [`DcsParams::from_strategy`].
//!
//! This module defines the value type only — wiring it into a context struct or
//! a pass is the responsibility of later DCS work.

use std::time::Instant;

use cobre_solver::{
    Basis, ProfiledSolver, RowBatch, SolutionView, SolverError, SolverInterface, StageTemplate,
};

use crate::basis_reconstruct::{
    ReconstructionTarget, enforce_basic_count_invariant, reconstruct_basis_uniform_basic,
};
use crate::cut::{CutPool, CutRowMap};
use crate::cut_selection::CutSelectionStrategy;
use crate::error::SddpError;
use crate::forward::append_slots_to_lp;
use crate::gemm::gemm_block;
use crate::indexer::StageIndexer;
use crate::workspace::CapturedBasis;

/// Dynamic Cut Selection hyperparameters.
///
/// A `Copy`, allocation-free value type. Field invariants (`nadic >= 1`,
/// `epsilon_viol > 0`, `k1` either `None` or `Some(>= 1)`) are enforced by the
/// config-parse step that produces
/// [`CutSelectionStrategy::Dynamic`](crate::cut_selection::CutSelectionStrategy::Dynamic);
/// `DcsParams` trusts its inputs and documents the invariants here.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DcsParams {
    /// Candidate-recency window.
    ///
    /// `None` ⇒ `∞`: every pool cut is a candidate. This is the
    /// exactness-preserving default. `Some(n)` ⇒ only cuts whose
    /// `iteration_generated` is within the last `n` SDDP iterations are
    /// candidates (windowed, deliberately non-exact). Consumed by the scoring
    /// kernel.
    pub k1: Option<u32>,

    /// Initial-set history window: how far back to seed the active set at the
    /// start of a lazy solve.
    pub k2: u32,

    /// Maximum number of cuts added per inner iteration (`>= 1`).
    pub nadic: u32,

    /// Violation tolerance for accepting a candidate cut (`> 0`, and at least
    /// the LP dual-feasibility tolerance).
    pub epsilon_viol: f64,

    /// First SDDP iteration (1-based) at which DCS is applied.
    pub start_iteration: u64,

    /// Bounded inner-loop cap: maximum number of lazy add-and-resolve rounds
    /// before falling back to a terminating-condition solve.
    pub max_inner_iterations: u32,
}

impl Default for DcsParams {
    fn default() -> Self {
        Self {
            k1: None,
            k2: 5,
            nadic: 10,
            epsilon_viol: 1e-10,
            start_iteration: 2,
            max_inner_iterations: 50,
        }
    }
}

impl DcsParams {
    /// Build [`DcsParams`] from a cut-selection strategy.
    ///
    /// Returns `Some` only for the
    /// [`Dynamic`](crate::cut_selection::CutSelectionStrategy::Dynamic) variant,
    /// copying its five fields (`k1`, `k2`, `nadic`, `epsilon_viol`,
    /// `start_iteration`) and taking `max_inner_iterations` from
    /// [`DcsParams::default`]. Returns `None` for every other variant.
    #[must_use]
    pub fn from_strategy(strategy: &CutSelectionStrategy) -> Option<Self> {
        match strategy {
            CutSelectionStrategy::Dynamic {
                k1,
                k2,
                nadic,
                epsilon_viol,
                start_iteration,
            } => Some(Self {
                k1: *k1,
                k2: *k2,
                nadic: *nadic,
                epsilon_viol: *epsilon_viol,
                start_iteration: *start_iteration,
                max_inner_iterations: Self::default().max_inner_iterations,
            }),
            CutSelectionStrategy::Level1 { .. }
            | CutSelectionStrategy::Lml1 { .. }
            | CutSelectionStrategy::Dominated { .. } => None,
        }
    }

    /// Return whether DCS is active at the given 1-based SDDP `iteration`.
    ///
    /// `true` when `iteration >= self.start_iteration`.
    #[must_use]
    pub fn is_active(&self, iteration: u64) -> bool {
        iteration >= self.start_iteration
    }
}

// ---------------------------------------------------------------------------
// DcsScoringScratch
// ---------------------------------------------------------------------------

/// Reusable scratch buffers for [`score_violated_candidates`].
///
/// Held across solves so the scoring kernel allocates nothing on the hot path.
/// Grow capacities once up front via [`DcsScoringScratch::reserve`]; the kernel
/// only clears and re-fills.
#[derive(Default)]
pub struct DcsScoringScratch {
    /// Unscaled (raw) state vector at the current LP optimum, one entry per
    /// state index (`indexer.n_state` long after fill).
    pub unscaled_state: Vec<f64>,

    /// Gathered coefficient rows of the eligible candidates for a single scoring
    /// pass, row-major `k_rows × n_state`. Cleared and re-filled each pass; the
    /// single batched [`gemm_block`] reads it as the GEMM left operand. Reserved
    /// to `pool_capacity * n_state` so even an all-eligible pass needs no growth.
    pub cand_coef_block: Vec<f64>,

    /// Per-candidate activity output of the single batched GEMM, one entry per
    /// gathered candidate (`alpha[i] = ∇·x*_raw` for candidate `i`, before the
    /// per-candidate intercept is added). Length tracks the gathered candidate
    /// count each pass.
    pub alpha: Vec<f64>,

    /// Slot id of each gathered candidate, parallel to the rows of
    /// `cand_coef_block` and (after the GEMM) to `alpha`. Lets the post-GEMM
    /// violation scan recover each activity entry's slot id (and, via
    /// `pool.intercepts`, its intercept).
    pub cand_slots: Vec<u32>,

    /// `(violation_magnitude, slot)` for each violated candidate, before the
    /// top-`nadic` truncation.
    pub violations: Vec<(f64, u32)>,
}

impl DcsScoringScratch {
    /// Grow the scratch buffers to hold `n_state` state entries and up to
    /// `pool_capacity` candidate cuts, without shrinking existing capacity
    /// (growth-only, mirroring `workspace.rs`).
    ///
    /// `cand_coef_block` is reserved to `pool_capacity * n_state` — the worst
    /// case where every active cut is an eligible (non-resident, in-window)
    /// candidate — while `alpha`, `cand_slots`, and `violations` are reserved to
    /// `pool_capacity` entries each.
    pub fn reserve(&mut self, n_state: usize, pool_capacity: usize) {
        if self.unscaled_state.capacity() < n_state {
            self.unscaled_state
                .reserve(n_state - self.unscaled_state.capacity());
        }
        let coef_capacity = pool_capacity * n_state;
        if self.cand_coef_block.capacity() < coef_capacity {
            self.cand_coef_block
                .reserve(coef_capacity - self.cand_coef_block.capacity());
        }
        if self.alpha.capacity() < pool_capacity {
            self.alpha.reserve(pool_capacity - self.alpha.capacity());
        }
        if self.cand_slots.capacity() < pool_capacity {
            self.cand_slots
                .reserve(pool_capacity - self.cand_slots.capacity());
        }
        if self.violations.capacity() < pool_capacity {
            self.violations
                .reserve(pool_capacity - self.violations.capacity());
        }
    }
}

// ---------------------------------------------------------------------------
// score_violated_candidates
// ---------------------------------------------------------------------------

/// Score the non-resident cut candidates at the current LP optimum and select
/// the most-violated `nadic` to add.
///
/// # Sign & scale contract
///
/// Cut coefficients are stored **raw** (`∂Q/∂x`); the LP solves in **scaled**
/// space. The cut row is `−∇·x + θ ≥ intercept` (see `.claude/rules/sddp.md`),
/// so the cut's `θ`-floor at the raw optimum `x*_raw` is
/// `alpha = intercept + ∇·x*_raw`. A candidate is **violated** when the current
/// LP `θ*_raw` falls below that floor by more than `epsilon_viol`, i.e.
/// `v = alpha − theta_raw > epsilon_viol` (strict).
///
/// Both `θ*` and every state column are unscaled by `col_scale` before scoring
/// (`x_raw = col_scale[c] · x_scaled`); mixing scaled state with raw
/// coefficients is the classic silent bug. An empty `col_scale` means no
/// scaling (factor 1.0).
///
/// State index → LP column mapping uses
/// [`StageIndexer::state_to_lp_column`], identical to the cut-row builder in
/// `forward::build_cut_row_batch_into`, so scoring and row construction
/// reference the same columns.
///
/// # Batched scoring
///
/// A single pass over [`CutPool::active_cuts`] (ascending-slot order) applies
/// the filters below and **gathers** each surviving candidate's `coefficients`
/// slice into the contiguous row-major `scratch.cand_coef_block`
/// (`k_rows × n_state`), recording its slot in `scratch.cand_slots`. One
/// [`gemm_block`] then fills `scratch.alpha[0..k_rows]` with every candidate's
/// `∇·x*_raw` in a single dispatch. Because `gemm_block` computes each output
/// row's dot product independently, a candidate's activity is **bit-identical**
/// whether scored alone (`k_rows = 1`) or in a batch (`k_rows = N`); the
/// per-candidate intercept (`pool.intercepts[slot]`) is added afterward, outside
/// the GEMM. The violation scan, sort, and `nadic` truncation are unchanged.
///
/// # `k1` candidate-recency window
///
/// Applied first. A cut at slot `s` is eligible only when `params.k1` is `None`
/// (∞ — every cut a candidate, the exactness path) or its age
/// `current_iteration.saturating_sub(pool.metadata[s].iteration_generated)` is
/// `< k1`. Warm-start cuts carry `iteration_generated == u64::MAX`, so their
/// saturating age is `0` and they are always inside any finite window. Cuts
/// outside the window are skipped entirely — never scored, never added.
///
/// # Selection & determinism
///
/// Violated candidates are sorted by violation **descending**, ties broken by
/// **ascending slot id** (via [`f64::total_cmp`], which is panic-free and
/// NaN-stable — never `partial_cmp().unwrap()`). The first `nadic` slot ids are
/// written to `out_selected` (cleared first). The choice is a deterministic
/// function of `(pool contents, x*, ε_viol, nadic, k1, current_iteration)`.
///
/// # Returns
///
/// The total count of violated candidates (≥ `out_selected.len()`). The caller
/// uses this to detect the no-violation stop and drive the TC fallback.
///
/// # Allocation
///
/// Allocation-free beyond growth of the pre-reserved `scratch` vectors. Call
/// [`DcsScoringScratch::reserve`] before the hot loop.
pub fn score_violated_candidates(
    pool: &CutPool,
    indexer: &StageIndexer,
    primal: &[f64],
    col_scale: &[f64],
    resident: &CutRowMap,
    params: &DcsParams,
    current_iteration: u64,
    scratch: &mut DcsScoringScratch,
    out_selected: &mut Vec<u32>,
) -> usize {
    let n_state = indexer.n_state;
    let theta = indexer.theta;

    // `primal.len() >= theta + 1` (i.e. the theta column is in range), written
    // as `> theta` to satisfy clippy::int_plus_one.
    debug_assert!(
        primal.len() > theta,
        "score_violated_candidates: primal.len() {} <= theta ({theta})",
        primal.len(),
    );
    debug_assert_eq!(
        pool.state_dimension, n_state,
        "score_violated_candidates: pool.state_dimension {} != indexer.n_state {}",
        pool.state_dimension, n_state,
    );
    // When scaling is active, `col_scale` is per-column and must cover every LP
    // column read below — both `col_scale[theta]` and `col_scale[c]` for each
    // state column `c = state_to_lp_column(j)`. It is sized like `primal`
    // (one entry per column), so a single length check covers both accesses.
    debug_assert!(
        col_scale.is_empty() || col_scale.len() == primal.len(),
        "score_violated_candidates: col_scale.len() {} != primal.len() {} (non-empty col_scale \
         must be per-column)",
        col_scale.len(),
        primal.len(),
    );

    out_selected.clear();
    scratch.violations.clear();
    scratch.cand_coef_block.clear();
    scratch.cand_slots.clear();
    scratch.alpha.clear();

    // Unscale θ*: scaled space ⇒ raw via col_scale[theta] (empty ⇒ factor 1.0).
    let theta_raw = if col_scale.is_empty() {
        primal[theta]
    } else {
        col_scale[theta] * primal[theta]
    };

    // Unscale the raw state vector at the LP state columns. The column mapping
    // mirrors `forward::build_cut_row_batch_into` exactly.
    scratch.unscaled_state.clear();
    for j in 0..n_state {
        let c = indexer.state_to_lp_column(j);
        let x_raw = if col_scale.is_empty() {
            primal[c]
        } else {
            col_scale[c] * primal[c]
        };
        scratch.unscaled_state.push(x_raw);
    }

    // Gather pass: iterate active cuts in ascending-slot order, apply the k1
    // window then the resident-skip, and copy each surviving candidate's
    // coefficient row into the contiguous `cand_coef_block` block (recording its
    // slot in `cand_slots`). The intercept is recovered later via
    // `pool.intercepts[slot]`, so it need not be gathered here.
    for (slot, _intercept, coefficients) in pool.active_cuts() {
        // k1 candidate-recency window (applied first). saturating_sub keeps
        // warm-start cuts (iteration_generated == u64::MAX, age 0) eligible.
        if let Some(k1) = params.k1 {
            let age = current_iteration.saturating_sub(pool.metadata[slot].iteration_generated);
            if age >= u64::from(k1) {
                continue;
            }
        }

        // Skip slots already resident in the LP.
        if resident.lp_row_for_slot(slot).is_some() {
            continue;
        }

        scratch.cand_coef_block.extend_from_slice(coefficients);
        #[allow(clippy::cast_possible_truncation)]
        scratch.cand_slots.push(slot as u32);
    }

    let k_rows = scratch.cand_slots.len();

    // One batched GEMM for the whole pass: alpha[i] = ∇_i · x*_raw for each
    // gathered candidate i (single trial point, m_len = 1). gemm_block computes
    // each output row independently, so each alpha is bit-identical to the
    // per-candidate (k_rows = 1) call it replaces. A zero candidate count is a
    // gemm_block no-op (it returns immediately for k_rows == 0).
    scratch.alpha.clear();
    scratch.alpha.resize(k_rows, 0.0);
    gemm_block(
        &scratch.cand_coef_block,
        &scratch.unscaled_state,
        k_rows,
        n_state,
        1,
        &mut scratch.alpha,
    );

    // Violation scan over the batched activities: v = (intercept + ∇·x*_raw) −
    // θ_raw, with the intercept recovered per candidate from `pool.intercepts`.
    for (i, &slot) in scratch.cand_slots.iter().enumerate() {
        let alpha = pool.intercepts[slot as usize] + scratch.alpha[i];
        let v = alpha - theta_raw;
        if v > params.epsilon_viol {
            scratch.violations.push((v, slot));
        }
    }

    let violated_count = scratch.violations.len();

    // Sort by violation descending, ties by ascending slot id. total_cmp is
    // panic-free and NaN-stable (no partial_cmp().unwrap()).
    scratch
        .violations
        .sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

    let take = (params.nadic as usize).min(violated_count);
    out_selected.extend(scratch.violations[..take].iter().map(|&(_, slot)| slot));

    violated_count
}

// ---------------------------------------------------------------------------
// DcsSolveContext
// ---------------------------------------------------------------------------

/// Per-(stage, solve) context for [`lazy_solve_preloaded`], mirroring the error-context
/// fields `run_stage_solve` uses.
#[derive(Clone, Copy, Debug)]
pub struct DcsSolveContext {
    /// Stage index `t`.
    pub stage_index: usize,
    /// Scenario index `m`.
    pub scenario_index: usize,
    /// Training iteration (1-based) feeding the `k1` recency window, or `None`
    /// off the training path (e.g. simulation). `None` is treated as `0`, which
    /// disables the window: `0.saturating_sub(iteration_generated) == 0 < k1`,
    /// so every cut is eligible.
    pub iteration: Option<u64>,

    /// Continue from a carried LP instead of starting a fresh solve.
    ///
    /// `false` (the default for every fresh solve): reset the scratch
    /// [`CutRowMap`], append `initial_resident`, and run the initial solve
    /// (warm if a `stored_basis` was supplied, else cold). This is the only mode
    /// the forward, simulation, and per-opening backward paths use.
    ///
    /// `true` (backward opening-reuse path, openings 1..): the solver already
    /// holds the previous opening's cut rows (tracked in the carried scratch
    /// `CutRowMap`) and a warm basis; only the opening's noise bounds changed.
    /// Skip the reset / `initial_resident` append / model reload entirely; just
    /// warm-`solve(None)` to re-optimize under the new bounds, then run the lazy
    /// loop, which appends only the cuts this opening additionally violates.
    /// `initial_resident` and `stored_basis` are ignored in this mode.
    pub continue_carry: bool,
}

// ---------------------------------------------------------------------------
// DcsSolveScratch
// ---------------------------------------------------------------------------

/// Reusable buffers for [`lazy_solve_preloaded`], held across solves so the
/// steady-state hot path allocates nothing.
///
/// These buffers are held outside `SolverWorkspace` so `lazy_solve_preloaded`
/// is testable in isolation.
pub struct DcsSolveScratch {
    /// Add-row construction buffer for `add_rows`.
    pub batch: RowBatch,
    /// Candidate-scoring scratch (see [`DcsScoringScratch`]).
    pub scoring: DcsScoringScratch,
    /// Selected slot ids from the most recent scoring call.
    pub out_selected: Vec<u32>,
    /// Destination basis for the initial uniform-BASIC reconstruction.
    pub recon_basis: Basis,
    /// Slot→LP-row map for the cut rows resident in the loaded LP.
    ///
    /// Held in scratch (rather than allocated per call) so the steady-state hot
    /// path does not allocate a fresh `vec![None; populated_count]` every solve,
    /// and so the backward opening-reuse path can carry residency across a trial
    /// point's openings: a fresh solve resets it (see
    /// [`DcsSolveContext::continue_carry`]), a continued solve leaves it intact.
    pub row_map: CutRowMap,
    /// Owned copy of the final solve's primal vector (one entry per LP column).
    ///
    /// [`lazy_solve_preloaded`] returns `Result<()>` rather than a borrowing
    /// `SolutionView`: on the terminating path it copies the live solve's
    /// solution into these caller-owned `res_*` buffers, and the caller rebuilds
    /// a zero-cost view over them via [`DcsSolveScratch::result_view`]. This
    /// avoids returning a loan obtained inside the lazy loop (rejected by stable
    /// NLL) without keeping the redundant exit re-solve.
    pub res_primal: Vec<f64>,
    /// Owned copy of the final solve's dual vector. On the DCS path this is the
    /// FULL dual — structural rows followed by the resident cut rows — and is
    /// copied verbatim (never truncated): the simulation reader relies on the
    /// structural-row prefix and ignores the trailing cut-row entries.
    pub res_dual: Vec<f64>,
    /// Owned copy of the final solve's reduced-cost vector (one entry per LP
    /// column).
    pub res_reduced_costs: Vec<f64>,
    /// Owned copy of the final solve's objective value.
    pub res_objective: f64,
    /// Owned copy of the final solve's simplex iteration count.
    pub res_iterations: u64,
    /// Owned copy of the final solve's wall-clock solve time, in seconds.
    pub res_solve_time_seconds: f64,
    /// Cumulative wall time, in seconds, spent inside candidate-scoring
    /// (`score_violated_candidates`) across every [`lazy_solve_preloaded`] call
    /// that used this scratch. Grows monotonically and is never reset by the
    /// solve loop, so a per-worker accumulator survives across all (stage,
    /// solve) pairs. Read by instrumentation after a run to compute the
    /// scoring-versus-solve split; pure measurement, off the steady-state hot
    /// path when no observer reads it.
    pub scoring_time_seconds: f64,
    /// Cumulative sum of the resident cut-row count (`row_map.total_cut_rows()`)
    /// over every solve that used this scratch — one term per
    /// [`lazy_solve_preloaded`] completion (see [`Self::store_result`]). With
    /// [`Self::rows_in_lp_count`] this yields the mean rows-in-LP per solve.
    /// Grows monotonically; never reset by the solve loop (a per-worker
    /// accumulator surviving across all (stage, solve) pairs, like
    /// [`Self::scoring_time_seconds`]). Pure measurement, off the steady-state
    /// hot path.
    pub rows_in_lp_sum: u64,
    /// Number of solves folded into [`Self::rows_in_lp_sum`] (one per
    /// [`Self::store_result`] call). The denominator for the mean.
    pub rows_in_lp_count: u64,
    /// Largest resident cut-row count observed across every solve that used this
    /// scratch.
    pub rows_in_lp_max: u64,
}

impl Default for DcsSolveScratch {
    fn default() -> Self {
        Self {
            batch: RowBatch {
                num_rows: 0,
                row_starts: Vec::new(),
                col_indices: Vec::new(),
                values: Vec::new(),
                row_lower: Vec::new(),
                row_upper: Vec::new(),
            },
            scoring: DcsScoringScratch::default(),
            out_selected: Vec::new(),
            recon_basis: Basis::new(0, 0),
            res_primal: Vec::new(),
            res_dual: Vec::new(),
            res_reduced_costs: Vec::new(),
            res_objective: 0.0,
            res_iterations: 0,
            res_solve_time_seconds: 0.0,
            scoring_time_seconds: 0.0,
            rows_in_lp_sum: 0,
            rows_in_lp_count: 0,
            rows_in_lp_max: 0,
            row_map: CutRowMap::new(0, 0),
        }
    }
}

impl DcsSolveScratch {
    /// Grow the inner buffers to fit `n_state` state entries and a pool of
    /// `pool_capacity` cuts, without shrinking (growth-only).
    pub fn reserve(&mut self, n_state: usize, pool_capacity: usize) {
        self.scoring.reserve(n_state, pool_capacity);
        if self.out_selected.capacity() < pool_capacity {
            self.out_selected
                .reserve(pool_capacity - self.out_selected.capacity());
        }
        // Pre-size the slot→row map so the first solve does not reallocate; the
        // `base_row_offset` here is a placeholder (each fresh solve resets it to
        // the loaded core's row count in `lazy_solve_preloaded`).
        self.row_map.reset(pool_capacity, 0);
        // Pre-grow the result buffers to a conservative non-zero capacity. Their
        // exact lengths (num_cols for primal/reduced_costs; structural rows + cut
        // rows for dual) depend on the loaded LP, not on these arguments, so the
        // first fill may still grow them; from then on `clear()` + `extend_from_slice`
        // reuses the warmed capacity (alloc-free steady state).
        for buf in [
            &mut self.res_primal,
            &mut self.res_dual,
            &mut self.res_reduced_costs,
        ] {
            if buf.capacity() < pool_capacity {
                buf.reserve(pool_capacity - buf.capacity());
            }
        }
    }

    /// Rebuild a zero-copy [`SolutionView`] over the result buffers filled by the
    /// most recent [`lazy_solve_preloaded`] call.
    ///
    /// The view borrows `self` immutably, so it composes with the other immutable
    /// reads a caller performs after the solve (e.g. the resident `row_map`). It
    /// is only meaningful immediately after a successful `lazy_solve_preloaded`
    /// call that filled the buffers; the slices are exactly the final solve's
    /// solution (the one the redundant exit re-solve used to recompute).
    #[must_use]
    pub fn result_view(&self) -> SolutionView<'_> {
        SolutionView {
            objective: self.res_objective,
            primal: &self.res_primal,
            dual: &self.res_dual,
            reduced_costs: &self.res_reduced_costs,
            iterations: self.res_iterations,
            solve_time_seconds: self.res_solve_time_seconds,
        }
    }

    /// Copy a live solve's solution into the reused result buffers.
    ///
    /// `clear()` + `extend_from_slice` keeps the steady state allocation-free
    /// (capacity persists across calls); the full slices are copied verbatim —
    /// the dual in particular is NOT truncated, so the simulation reader still
    /// sees the structural-row prefix it depends on.
    fn store_result(&mut self, view: &SolutionView<'_>) {
        self.res_objective = view.objective;
        self.res_iterations = view.iterations;
        self.res_solve_time_seconds = view.solve_time_seconds;
        self.res_primal.clear();
        self.res_primal.extend_from_slice(view.primal);
        self.res_dual.clear();
        self.res_dual.extend_from_slice(view.dual);
        self.res_reduced_costs.clear();
        self.res_reduced_costs.extend_from_slice(view.reduced_costs);

        // Per-solve resident-set size (rows-in-LP): the number of cut rows
        // resident in the LP at this solve's completion. Folded here because
        // `store_result` is the single completion point of every lazy solve
        // (both the exact-optimum and TC-fallback exits). Mirrors the
        // `scoring_time_seconds` accumulator — cumulative, never reset.
        let rows_in_lp = self.row_map.total_cut_rows() as u64;
        self.rows_in_lp_sum += rows_in_lp;
        self.rows_in_lp_count += 1;
        self.rows_in_lp_max = self.rows_in_lp_max.max(rows_in_lp);
    }
}

// ---------------------------------------------------------------------------
// lazy_solve_preloaded
// ---------------------------------------------------------------------------

/// Map a [`SolverError`] to an [`SddpError`] with stage/scenario/iteration
/// context — the same contract `stage_solve::run_stage_solve` uses
/// (`Infeasible` carries the context; everything else is `SddpError::Solver`).
fn map_solver_error(e: SolverError, ctx: DcsSolveContext) -> SddpError {
    match e {
        SolverError::Infeasible => SddpError::Infeasible {
            stage: ctx.stage_index,
            iteration: ctx.iteration.unwrap_or(0),
            scenario: ctx.scenario_index,
        },
        other => SddpError::Solver(other),
    }
}

/// Solve one (stage, solve) lazily under Dynamic Cut Selection, given an
/// already-loaded core LP.
///
/// **The caller owns the model load and any bounds patch.** Before calling,
/// the caller must have run `solver.load_model(core)` (the cut-free core LP)
/// and applied every per-solve bound (state pinning, opening noise, etc.) on
/// that loaded model. This routine does NOT call `load_model`; it only appends
/// cut rows and solves. `core` is consulted for dimensions only
/// (`num_rows` → cut-row offset and reconstruction `base_row_count`;
/// `num_cols` → reconstruction `num_cols`). Because every cut row is appended
/// to the loaded, already-patched LP and every inner re-solve is a warm
/// `solve(None)` (never another `load_model`), the caller's bound patch
/// survives the entire loop.
///
/// Pass-agnostic: it builds the cut set incrementally and runs the lazy inner
/// loop. On the terminating path it copies the final solve's solution into the
/// caller-owned result buffers in `scratch` and returns `Ok(())`; the caller
/// rebuilds a zero-cost [`SolutionView`] over those buffers via
/// [`DcsSolveScratch::result_view`]. Returning by buffer rather than by borrowing
/// view is what lets the no-violation arm avoid a redundant exit re-solve without
/// returning a loan obtained inside the loop (which stable NLL rejects). Post-solve
/// extraction (backward = dual, forward/simulation = primal) is the caller's job
/// and is NOT done here.
///
/// # Algorithm
///
/// 1. Build a fresh `CutRowMap::new(pool.populated_count, core.num_rows)` for
///    the already-loaded core.
/// 2. Append `initial_resident` (active, not-yet-resident slots) via
///    [`append_slots_to_lp`].
/// 3. Seed the warm basis: if `stored_basis` is `Some`, reconstruct it
///    uniform-BASIC ([`reconstruct_basis_uniform_basic`]), repair the
///    basic-count invariant ([`enforce_basic_count_invariant`]), and
///    `solve(Some(..))`; else `solve(None)` (cold).
/// 4. Inner loop, up to `params.max_inner_iterations`: score the omitted cuts
///    at the live `view.primal` via [`score_violated_candidates`] (using
///    `ctx.iteration.unwrap_or(0)` for the `k1` window); when none is violated
///    (the **exact** optimum) copy the live view into the `res_*` buffers and
///    return — **no re-solve**; else add the top-`nadic` violated slots, drop the
///    view, and warm-`solve(None)` again.
/// 5. **TC fallback**: if the cap is hit with violations remaining, add **all**
///    remaining violated candidates and solve once (the LP changed, so this final
///    solve is legitimate), copy that view into the `res_*` buffers, and return —
///    preserving exactness.
///
/// A cold mid-loop re-solve (a `solve(None)` whose retry ladder discarded the
/// warm basis) is normal and tolerated — the loop never reads a stale basis
/// between solves.
///
/// # Errors
///
/// Propagates solver failures via the same mapping as `run_stage_solve`:
/// `SolverError::Infeasible` → [`SddpError::Infeasible`] with stage/scenario/
/// iteration context; every other variant → [`SddpError::Solver`].
///
/// # Allocation
///
/// Steady-state allocation-free: all working buffers come from `scratch` (grow
/// once via [`DcsSolveScratch::reserve`]); the result copy reuses the `res_*`
/// buffers via `clear()` + `extend_from_slice`. The per-call `CutRowMap`
/// allocation mirrors the per-(stage, solve) reset and is acceptable here.
// The argument list is the spec-mandated signature: each item is a distinct
// read-only input or scratch buffer with no natural grouping (the context-struct
// rule already bundles the per-(stage, solve) scalars into `ctx`/`params`).
#[allow(clippy::too_many_arguments)]
pub fn lazy_solve_preloaded<S: SolverInterface>(
    solver: &mut ProfiledSolver<S>,
    core: &StageTemplate,
    pool: &CutPool,
    indexer: &StageIndexer,
    col_scale: &[f64],
    stored_basis: Option<&CapturedBasis>,
    initial_resident: &[u32],
    params: &DcsParams,
    scratch: &mut DcsSolveScratch,
    ctx: DcsSolveContext,
) -> Result<(), SddpError> {
    let current_iteration = ctx.iteration.unwrap_or(0);

    // Steps 1-3: prepare the LP and run the initial solve, producing the live
    // `view` the lazy loop holds. Two modes (see `DcsSolveContext::continue_carry`).
    let mut view = if ctx.continue_carry {
        // CONTINUE (backward opening-reuse, openings 1..): the solver already
        // holds the previous opening's cut rows (tracked in `scratch.row_map`)
        // and a warm basis; the caller patched only the new opening's bounds. Do
        // NOT reset the map, append the seed, or reload — just re-solve warm to
        // re-optimize under the new bounds and fall through to the lazy loop,
        // which appends only the cuts this opening additionally violates.
        // `initial_resident` and `stored_basis` are ignored in this mode.
        solver.solve(None).map_err(|e| map_solver_error(e, ctx))?
    } else {
        // FRESH: reset the carried row map for the already-loaded core (the
        // caller has run `load_model(core)` and applied any bounds patch; this
        // routine must NOT reload — that would discard the patch), append the
        // initial resident subset, then run the initial solve (warm uniform-
        // BASIC if a stored basis exists, else cold).
        scratch.row_map.reset(pool.populated_count, core.num_rows);
        append_slots_to_lp(
            solver,
            pool,
            initial_resident,
            indexer,
            col_scale,
            &mut scratch.row_map,
            &mut scratch.batch,
        );

        let cut_rows = scratch.row_map.total_cut_rows();
        if let Some(stored) = stored_basis {
            let target = ReconstructionTarget {
                base_row_count: core.num_rows,
                num_cols: core.num_cols,
            };
            reconstruct_basis_uniform_basic(stored, target, cut_rows, &mut scratch.recon_basis);
            enforce_basic_count_invariant(
                &mut scratch.recon_basis,
                core.num_rows + cut_rows,
                core.num_rows,
            );
            solver
                .solve(Some(&scratch.recon_basis))
                .map_err(|e| map_solver_error(e, ctx))?
        } else {
            solver.solve(None).map_err(|e| map_solver_error(e, ctx))?
        }
    };

    // Step 4/5: bounded lazy inner loop with TC fallback. The loop body scores
    // the LIVE `view.primal` in place (no copy): `score_violated_candidates`
    // borrows `scratch.scoring`/`scratch.row_map`, not the solver, so it composes
    // with the held view. On the no-violation arm it copies the live view into the
    // `res_*` buffers and returns `Ok(())` — NO redundant re-solve. On the add arm
    // it drops the view, appends the selected slots, and re-solves to refresh it.
    //
    // Because the function returns `Result<()>` (not a borrowing view), `view`
    // never escapes the loop, so holding it across the iteration is accepted by
    // stable NLL — the Polonius "return a loan from a loop" gap is sidestepped.
    for _ in 0..params.max_inner_iterations {
        // Cumulative scoring-time instrumentation: `scratch.scoring` (the inner
        // scoring buffers) is borrowed during the call; the `+=` on
        // `scratch.scoring_time_seconds` runs only after the call returns and
        // touches a distinct field, so there is no borrow conflict. `view.primal`
        // borrows the solver, which `score_violated_candidates` does not touch.
        let t0 = Instant::now();
        let violated = score_violated_candidates(
            pool,
            indexer,
            view.primal,
            col_scale,
            &scratch.row_map,
            params,
            current_iteration,
            &mut scratch.scoring,
            &mut scratch.out_selected,
        );
        scratch.scoring_time_seconds += t0.elapsed().as_secs_f64();

        if violated == 0 {
            // Exact: the current resident subset reproduces the all-cuts optimum
            // and the live view already holds it. Copy it into the result buffers
            // and return — the LP is unchanged and re-solving would recompute the
            // same point.
            scratch.store_result(&view);
            return Ok(());
        }

        // A cut will be added, so the held view is stale: end its solver borrow
        // here (`SolutionView` is `Copy`, so the borrow lasts only to its last
        // use), then append the selected top-`nadic` slots and re-solve (warm;
        // may escalate to cold — never assume the warm basis survived).
        let _ = view;
        append_slots_to_lp(
            solver,
            pool,
            &scratch.out_selected,
            indexer,
            col_scale,
            &mut scratch.row_map,
            &mut scratch.batch,
        );
        view = solver.solve(None).map_err(|e| map_solver_error(e, ctx))?;
    }

    // TC fallback: the cap was hit with violations still present. Collect ALL
    // remaining violated candidates (effective nadic = ∞) from the live primal,
    // add them, and solve once. This degrades to a terminating-condition
    // (all-cuts) solve for this (stage, solve) but preserves exactness.
    let mut all_params = *params;
    all_params.nadic = u32::MAX;
    // Same cumulative-scoring instrumentation as the inner loop above.
    let t0 = Instant::now();
    let remaining = score_violated_candidates(
        pool,
        indexer,
        view.primal,
        col_scale,
        &scratch.row_map,
        &all_params,
        current_iteration,
        &mut scratch.scoring,
        &mut scratch.out_selected,
    );
    scratch.scoring_time_seconds += t0.elapsed().as_secs_f64();
    // The live view's solver borrow must end before the appending re-solve
    // (`SolutionView` is `Copy`, so the borrow lasts only to this last use).
    let _ = view;
    if remaining > 0 {
        append_slots_to_lp(
            solver,
            pool,
            &scratch.out_selected,
            indexer,
            col_scale,
            &mut scratch.row_map,
            &mut scratch.batch,
        );
    }
    let view = solver.solve(None).map_err(|e| map_solver_error(e, ctx))?;
    scratch.store_result(&view);
    Ok(())
}

// ---------------------------------------------------------------------------
// build_initial_resident_set
// ---------------------------------------------------------------------------

/// Build the DCS initial resident-cut set from synchronized pool metadata.
///
/// The initial resident set is the warm-start hint [`lazy_solve_preloaded`] consumes as
/// `initial_resident`. By exactness the converged optimum is independent of it,
/// but the seed must be a deterministic function of the pool's per-slot
/// metadata only — never of any solve trace, worker id, or MPI rank — so that
/// results are invariant to the rank count. It reads only `last_active_iter`
/// and `iteration_generated`, both maintained deterministically each iteration
/// from the MPI-gathered visited states, so seeding from them preserves that
/// invariance.
///
/// Clears `out`, then for each populated slot `s` in ascending order
/// (`0..pool.populated_count`) includes `s` iff `pool.active[s]` AND either:
///
/// - the slot was active within the last `k2` iterations
///   (`current_iteration.saturating_sub(pool.metadata[s].last_active_iter) <= k2`),
///   or
/// - the slot was generated in the current iteration
///   (`pool.metadata[s].iteration_generated == current_iteration`) — these cuts
///   have not been tested yet and are always seeded (mirroring the
///   current-iteration protection in `cut_selection`).
///
/// `saturating_sub` is required because `last_active_iter` can exceed
/// `current_iteration` for current-iteration cuts; plain subtraction would
/// underflow.
///
/// Ascending slot order makes the result deterministic and declaration-order
/// invariant. Allocates nothing beyond growth of `out` (the caller pre-reserves
/// to `pool.populated_count`).
pub fn build_initial_resident_set(
    pool: &CutPool,
    current_iteration: u64,
    k2: u32,
    out: &mut Vec<u32>,
) {
    debug_assert!(
        pool.metadata.len() >= pool.populated_count,
        "build_initial_resident_set: metadata.len() {} < populated_count {}",
        pool.metadata.len(),
        pool.populated_count,
    );

    out.clear();
    let window = u64::from(k2);
    #[allow(clippy::cast_possible_truncation)]
    for s in 0..pool.populated_count {
        if !pool.active[s] {
            continue;
        }
        let meta = &pool.metadata[s];
        let within_window = current_iteration.saturating_sub(meta.last_active_iter) <= window;
        let is_current_iter = meta.iteration_generated == current_iteration;
        if within_window || is_current_iter {
            out.push(s as u32);
        }
    }
}

#[cfg(test)]
#[allow(clippy::doc_markdown)]
mod tests {
    use cobre_solver::{
        ActiveProfile, ActiveSolver, Basis, ProfiledSolver, RowBatch, SolverError, SolverInterface,
        SolverStatistics, StageTemplate,
    };

    use super::{
        DcsParams, DcsScoringScratch, DcsSolveContext, DcsSolveScratch, build_initial_resident_set,
        lazy_solve_preloaded, score_violated_candidates,
    };
    use crate::cut::{CutPool, CutRowMap};
    use crate::cut_selection::{CutMetadata, CutSelectionStrategy};
    use crate::indexer::StageIndexer;

    #[test]
    fn default_matches_spec() {
        let params = DcsParams::default();
        assert_eq!(
            params,
            DcsParams {
                k1: None,
                k2: 5,
                nadic: 10,
                epsilon_viol: 1e-10,
                start_iteration: 2,
                max_inner_iterations: 50,
            }
        );
        assert_eq!(params.k1, None);
    }

    #[test]
    fn from_strategy_dynamic_copies_fields() {
        // k1 = Some(n)
        let strategy = CutSelectionStrategy::Dynamic {
            k1: Some(20),
            k2: 7,
            nadic: 3,
            epsilon_viol: 1e-9,
            start_iteration: 4,
        };
        let params = DcsParams::from_strategy(&strategy)
            .expect("from_strategy must return Some for the Dynamic variant");
        assert_eq!(
            params,
            DcsParams {
                k1: Some(20),
                k2: 7,
                nadic: 3,
                epsilon_viol: 1e-9,
                start_iteration: 4,
                max_inner_iterations: 50,
            }
        );

        // k1 = None passes through as None.
        let strategy_inf = CutSelectionStrategy::Dynamic {
            k1: None,
            k2: 5,
            nadic: 10,
            epsilon_viol: 1e-10,
            start_iteration: 2,
        };
        let params_inf = DcsParams::from_strategy(&strategy_inf)
            .expect("from_strategy must return Some for the Dynamic variant");
        assert_eq!(params_inf.k1, None);
    }

    #[test]
    fn from_strategy_non_dynamic_is_none() {
        let level1 = CutSelectionStrategy::Level1 {
            check_frequency: 5,
            tie_tolerance: 1e-10,
        };
        assert!(DcsParams::from_strategy(&level1).is_none());

        let dominated = CutSelectionStrategy::Dominated {
            threshold: 1e-6,
            check_frequency: 10,
        };
        assert!(DcsParams::from_strategy(&dominated).is_none());
    }

    #[test]
    fn is_active_threshold() {
        let params = DcsParams {
            start_iteration: 2,
            ..DcsParams::default()
        };
        assert_eq!(
            [
                params.is_active(0),
                params.is_active(1),
                params.is_active(2),
                params.is_active(3),
            ],
            [false, false, true, true]
        );
    }

    #[test]
    fn dcs_params_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<DcsParams>();
    }

    // -----------------------------------------------------------------------
    // score_violated_candidates fixtures
    // -----------------------------------------------------------------------

    // All scoring tests use n_state = 2 (StageIndexer::new(2, 0)):
    //   - state columns 0, 1 (identity state_to_lp_column for j < hydro_count)
    //   - theta column 6 (= n * (3 + l) with n = 2, l = 0)
    // So `primal` must be at least length 7.
    const N_STATE: usize = 2;
    const THETA_COL: usize = 6;
    const PRIMAL_LEN: usize = THETA_COL + 1;

    fn indexer() -> StageIndexer {
        StageIndexer::new(2, 0)
    }

    /// A pool with capacity 16, state_dimension 2, forward_passes 16, no
    /// warm-start. `add_cut(0, slot, ..)` then maps to slot `slot`.
    fn empty_pool() -> CutPool {
        CutPool::new(16, N_STATE, 16, 0)
    }

    /// Insert a cut at `slot` with `intercept` and `coeffs`, and set its
    /// `iteration_generated`. Returns nothing; mutates `pool`.
    fn add(pool: &mut CutPool, slot: u32, intercept: f64, coeffs: &[f64], iter_generated: u64) {
        pool.add_cut(0, slot, intercept, coeffs);
        pool.metadata[slot as usize].iteration_generated = iter_generated;
    }

    /// An empty resident map (no slot is in the LP). base_row_offset is
    /// irrelevant to `lp_row_for_slot`, which returns `None` for every slot.
    fn empty_resident() -> CutRowMap {
        CutRowMap::new(16, 0)
    }

    fn params(nadic: u32, epsilon_viol: f64, k1: Option<u32>) -> DcsParams {
        DcsParams {
            k1,
            nadic,
            epsilon_viol,
            ..DcsParams::default()
        }
    }

    // -----------------------------------------------------------------------
    // score_violated_candidates tests
    // -----------------------------------------------------------------------

    /// AC1: two non-resident active cuts, violations 5.0 (slot A) and 2.0
    /// (slot B). With coeffs [0,0] and theta_raw = 0, alpha == intercept, so
    /// the intercepts ARE the violations. nadic = 10 → both selected, ordered
    /// descending.
    #[test]
    fn scores_and_orders_two_violated_descending() {
        let idx = indexer();
        let mut pool = empty_pool();
        add(&mut pool, 0, 5.0, &[0.0, 0.0], 1); // slot 0: violation 5.0
        add(&mut pool, 1, 2.0, &[0.0, 0.0], 1); // slot 1: violation 2.0
        let primal = vec![0.0_f64; PRIMAL_LEN]; // x* = 0, theta_raw = 0
        let resident = empty_resident();
        let p = params(10, 1e-10, None);
        let mut scratch = DcsScoringScratch::default();
        let mut out = Vec::new();

        let count = score_violated_candidates(
            &pool,
            &idx,
            &primal,
            &[],
            &resident,
            &p,
            10,
            &mut scratch,
            &mut out,
        );

        assert_eq!(count, 2);
        assert_eq!(
            out,
            vec![0, 1],
            "descending violation: slot 0 (5) then slot 1 (2)"
        );
    }

    /// AC2: same pool, nadic = 1. Full violated count is still 2, but only the
    /// top-1 slot is selected.
    #[test]
    fn respects_nadic_cap_returns_full_violated_count() {
        let idx = indexer();
        let mut pool = empty_pool();
        add(&mut pool, 0, 5.0, &[0.0, 0.0], 1);
        add(&mut pool, 1, 2.0, &[0.0, 0.0], 1);
        let primal = vec![0.0_f64; PRIMAL_LEN];
        let resident = empty_resident();
        let p = params(1, 1e-10, None);
        let mut scratch = DcsScoringScratch::default();
        let mut out = Vec::new();

        let count = score_violated_candidates(
            &pool,
            &idx,
            &primal,
            &[],
            &resident,
            &p,
            10,
            &mut scratch,
            &mut out,
        );

        assert_eq!(count, 2, "full violated count regardless of nadic");
        assert_eq!(out, vec![0], "top-1 only");
    }

    /// AC3: equal violation 3.0 at slots 7 and 4, nadic = 1 → ascending slot id
    /// tie-break selects slot 4.
    #[test]
    fn tie_break_ascending_slot_id() {
        let idx = indexer();
        let mut pool = empty_pool();
        add(&mut pool, 7, 3.0, &[0.0, 0.0], 1);
        add(&mut pool, 4, 3.0, &[0.0, 0.0], 1);
        let primal = vec![0.0_f64; PRIMAL_LEN];
        let resident = empty_resident();
        let p = params(1, 1e-10, None);
        let mut scratch = DcsScoringScratch::default();
        let mut out = Vec::new();

        let count = score_violated_candidates(
            &pool,
            &idx,
            &primal,
            &[],
            &resident,
            &p,
            10,
            &mut scratch,
            &mut out,
        );

        assert_eq!(count, 2);
        assert_eq!(out, vec![4], "equal violation → ascending slot id wins");
    }

    /// AC4: col_scale sign-flip. col_scale[theta] = 2.0, state-column scale 0.5.
    /// One cut: intercept 0.5, coeff [1.0, 0.0]. Scaled primal: x_scaled[0] = 2.0,
    /// theta_scaled = 1.0.
    ///   raw:    x_raw[0] = 0.5 * 2.0 = 1.0; theta_raw = 2.0 * 1.0 = 2.0.
    ///           alpha = 0.5 + 1.0*1.0 = 1.5; v = 1.5 - 2.0 = -0.5  → NOT violated.
    ///   if (buggy) NO unscaling: x[0] = 2.0; theta = 1.0.
    ///           alpha = 0.5 + 1.0*2.0 = 2.5; v = 2.5 - 1.0 = 1.5  → violated.
    /// The correct (unscaled) verdict is "not violated"; the verdict flips sign
    /// vs the unscaled-input bug.
    #[test]
    fn applies_col_scale_unscaling() {
        let idx = indexer();
        let mut pool = empty_pool();
        add(&mut pool, 0, 0.5, &[1.0, 0.0], 1);
        let mut primal = vec![0.0_f64; PRIMAL_LEN];
        primal[0] = 2.0; // scaled x[0]
        primal[THETA_COL] = 1.0; // scaled theta
        // col_scale must cover all columns up to theta (index 6).
        let mut col_scale = vec![1.0_f64; PRIMAL_LEN];
        col_scale[0] = 0.5; // state column 0 scale
        col_scale[THETA_COL] = 2.0; // theta scale
        let resident = empty_resident();
        let p = params(10, 1e-10, None);
        let mut scratch = DcsScoringScratch::default();
        let mut out = Vec::new();

        let count = score_violated_candidates(
            &pool,
            &idx,
            &primal,
            &col_scale,
            &resident,
            &p,
            10,
            &mut scratch,
            &mut out,
        );

        assert_eq!(
            count, 0,
            "with correct col_scale unscaling, v = -0.5 → not violated"
        );
        assert!(out.is_empty());
    }

    /// AC5: a resident slot is never selected, even when violated.
    #[test]
    fn skips_resident_slots() {
        let idx = indexer();
        let mut pool = empty_pool();
        add(&mut pool, 0, 5.0, &[0.0, 0.0], 1); // violated, but will be resident
        add(&mut pool, 1, 2.0, &[0.0, 0.0], 1); // violated, non-resident
        let primal = vec![0.0_f64; PRIMAL_LEN];
        let mut resident = empty_resident();
        resident.insert(0); // slot 0 is in the LP
        let p = params(10, 1e-10, None);
        let mut scratch = DcsScoringScratch::default();
        let mut out = Vec::new();

        let count = score_violated_candidates(
            &pool,
            &idx,
            &primal,
            &[],
            &resident,
            &p,
            10,
            &mut scratch,
            &mut out,
        );

        assert_eq!(count, 1, "resident slot 0 is excluded from scoring");
        assert_eq!(out, vec![1]);
    }

    /// AC6: k1 window. Two violated cuts: slot 0 generated at iteration 8
    /// (age 2), slot 1 generated at iteration 2 (age 8). current_iteration = 10,
    /// k1 = Some(5) → only the age-2 cut (slot 0) is in window. With k1 = None
    /// both are eligible.
    #[test]
    fn k1_window_filters_old_cuts() {
        let idx = indexer();
        let mut pool = empty_pool();
        add(&mut pool, 0, 5.0, &[0.0, 0.0], 8); // age 2 at iter 10
        add(&mut pool, 1, 2.0, &[0.0, 0.0], 2); // age 8 at iter 10
        let primal = vec![0.0_f64; PRIMAL_LEN];
        let resident = empty_resident();
        let mut scratch = DcsScoringScratch::default();
        let mut out = Vec::new();

        // Finite window k1 = 5: slot 1 (age 8 >= 5) skipped entirely.
        let p_finite = params(10, 1e-10, Some(5));
        let count_finite = score_violated_candidates(
            &pool,
            &idx,
            &primal,
            &[],
            &resident,
            &p_finite,
            10,
            &mut scratch,
            &mut out,
        );
        assert_eq!(count_finite, 1, "only the in-window cut is counted");
        assert_eq!(out, vec![0]);

        // k1 = None: exactness path, both eligible.
        let p_inf = params(10, 1e-10, None);
        let count_inf = score_violated_candidates(
            &pool,
            &idx,
            &primal,
            &[],
            &resident,
            &p_inf,
            10,
            &mut scratch,
            &mut out,
        );
        assert_eq!(count_inf, 2, "k1 = None ⇒ all cuts eligible");
        assert_eq!(out, vec![0, 1]);
    }

    /// k1 window keeps warm-start cuts (iteration_generated == u64::MAX) inside
    /// any finite window via saturating_sub (age 0).
    #[test]
    fn k1_window_keeps_warm_start_cuts() {
        let idx = indexer();
        let mut pool = empty_pool();
        add(&mut pool, 0, 5.0, &[0.0, 0.0], u64::MAX); // warm-start sentinel
        let primal = vec![0.0_f64; PRIMAL_LEN];
        let resident = empty_resident();
        let p = params(10, 1e-10, Some(1)); // tightest finite window
        let mut scratch = DcsScoringScratch::default();
        let mut out = Vec::new();

        let count = score_violated_candidates(
            &pool,
            &idx,
            &primal,
            &[],
            &resident,
            &p,
            10,
            &mut scratch,
            &mut out,
        );

        assert_eq!(count, 1, "warm-start cut (age 0) is always in window");
        assert_eq!(out, vec![0]);
    }

    /// AC7: violation exactly equal to epsilon_viol is NOT counted (strict >).
    #[test]
    fn epsilon_viol_is_strict() {
        let idx = indexer();
        let mut pool = empty_pool();
        // alpha = 1.0, theta_raw = 0 → v = 1.0; epsilon_viol = 1.0 → v == eps.
        add(&mut pool, 0, 1.0, &[0.0, 0.0], 1);
        let primal = vec![0.0_f64; PRIMAL_LEN];
        let resident = empty_resident();
        let p = params(10, 1.0, None);
        let mut scratch = DcsScoringScratch::default();
        let mut out = Vec::new();

        let count = score_violated_candidates(
            &pool,
            &idx,
            &primal,
            &[],
            &resident,
            &p,
            10,
            &mut scratch,
            &mut out,
        );

        assert_eq!(count, 0, "v == epsilon_viol is NOT strictly greater");
        assert!(out.is_empty());
    }

    /// No candidate violated → empty out_selected, return 0.
    #[test]
    fn no_violation_returns_empty() {
        let idx = indexer();
        let mut pool = empty_pool();
        // alpha = -1.0 < theta_raw = 0 → v = -1.0, not violated.
        add(&mut pool, 0, -1.0, &[0.0, 0.0], 1);
        let primal = vec![0.0_f64; PRIMAL_LEN];
        let resident = empty_resident();
        let p = params(10, 1e-10, None);
        let mut scratch = DcsScoringScratch::default();
        let mut out = Vec::new();

        let count = score_violated_candidates(
            &pool,
            &idx,
            &primal,
            &[],
            &resident,
            &p,
            10,
            &mut scratch,
            &mut out,
        );

        assert_eq!(count, 0);
        assert!(out.is_empty());
    }

    // -----------------------------------------------------------------------
    // Batched-scoring tests (ticket-018): bit-identical vs per-row reference,
    // single GEMM per pass, growth-only scratch.
    // -----------------------------------------------------------------------

    use crate::gemm::gemm_block;

    /// Deterministic splitmix64 PRNG (no external rand dep in unit tests), with a
    /// helper that draws a finite f64 in roughly [-1.5, 0.5). Mirrors the
    /// `benches/cut_selection_kernel.rs` generator so the randomized fixture is
    /// reproducible.
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn draw_f64(state: &mut u64) -> f64 {
        let r = splitmix64(state);
        let bits = (r >> 12) & ((1_u64 << 52) - 1);
        f64::from_bits((1023_u64 << 52) | bits) - 1.5
    }

    /// Per-row reference for the batched scorer: replays the EXACT filter / gather
    /// order of `score_violated_candidates`, but scores each surviving candidate
    /// with its own `gemm_block(coef, x*, 1, n_state, 1, ..)` call (the old
    /// per-candidate path). Returns `(alpha_bits_in_gather_order, out_selected)`.
    #[allow(clippy::cast_possible_truncation)]
    fn per_row_reference(
        pool: &CutPool,
        idx: &StageIndexer,
        primal: &[f64],
        col_scale: &[f64],
        resident: &CutRowMap,
        p: &DcsParams,
        current_iteration: u64,
    ) -> (Vec<u64>, Vec<u32>) {
        let n_state = idx.n_state;
        let theta = idx.theta;
        let theta_raw = if col_scale.is_empty() {
            primal[theta]
        } else {
            col_scale[theta] * primal[theta]
        };
        let mut unscaled_state = Vec::with_capacity(n_state);
        for j in 0..n_state {
            let c = idx.state_to_lp_column(j);
            let x_raw = if col_scale.is_empty() {
                primal[c]
            } else {
                col_scale[c] * primal[c]
            };
            unscaled_state.push(x_raw);
        }

        let mut alpha_bits = Vec::new();
        let mut violations: Vec<(f64, u32)> = Vec::new();
        for (slot, intercept, coefficients) in pool.active_cuts() {
            if let Some(k1) = p.k1 {
                let age = current_iteration.saturating_sub(pool.metadata[slot].iteration_generated);
                if age >= u64::from(k1) {
                    continue;
                }
            }
            if resident.lp_row_for_slot(slot).is_some() {
                continue;
            }
            let mut dot = [0.0_f64; 1];
            gemm_block(coefficients, &unscaled_state, 1, n_state, 1, &mut dot);
            // Record the GEMM's raw activity (∇·x*_raw), matching what the
            // batched scorer stores in `scratch.alpha` (intercept added after).
            alpha_bits.push(dot[0].to_bits());
            let alpha = intercept + dot[0];
            let v = alpha - theta_raw;
            if v > p.epsilon_viol {
                violations.push((v, slot as u32));
            }
        }
        violations.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let take = (p.nadic as usize).min(violations.len());
        let out_selected = violations[..take].iter().map(|&(_, s)| s).collect();
        (alpha_bits, out_selected)
    }

    /// Build a randomized capacity-`cap`, state-dim-2 pool: every slot active,
    /// random intercept + coefficients, random `iteration_generated` in
    /// `[1, 12]`. Deterministic in `seed`.
    #[allow(clippy::cast_possible_truncation)]
    fn random_pool(cap: usize, seed: u64) -> CutPool {
        let mut pool = CutPool::new(cap, N_STATE, cap as u32, 0);
        let mut state = seed;
        for slot in 0..cap {
            let intercept = draw_f64(&mut state);
            let coeffs = [draw_f64(&mut state), draw_f64(&mut state)];
            pool.add_cut(0, slot as u32, intercept, &coeffs);
            // iteration_generated in [1, 12].
            let gen_iter = 1 + (splitmix64(&mut state) % 12);
            pool.metadata[slot].iteration_generated = gen_iter;
        }
        pool
    }

    /// AC1: the batched scorer's per-candidate activities (`scratch.alpha`, by
    /// bits) and its `out_selected` are bit-identical to a per-row reference that
    /// scores each candidate with an individual `gemm_block(.., 1, n_state, 1, ..)`
    /// call — over a randomized pool with a non-trivial primal and col_scale, a
    /// mix of resident / non-resident slots, and a finite k1 window.
    #[test]
    fn batched_scoring_bit_identical_to_per_row_reference() {
        let idx = indexer();
        let cap = 64;
        let pool = random_pool(cap, 0x0BAD_F00D_C0FF_EE11);

        // Non-trivial primal: scaled x and theta drawn from the same PRNG stream.
        let mut prng = 0xDEAD_BEEF_1234_5678_u64;
        let mut primal = vec![0.0_f64; PRIMAL_LEN];
        for v in &mut primal {
            *v = draw_f64(&mut prng);
        }
        // Per-column scale covering every column up to theta.
        let mut col_scale = vec![1.0_f64; PRIMAL_LEN];
        for v in &mut col_scale {
            *v = 0.5 + (draw_f64(&mut prng) + 1.5) * 0.5; // strictly positive
        }
        // Drive theta_raw strongly negative so most candidates are violated
        // (v = alpha - theta_raw > 0): this makes the selection/sort/nadic path a
        // non-vacuous witness, not just the all-empty case. col_scale[theta] > 0,
        // so a negative scaled theta yields a negative theta_raw.
        primal[THETA_COL] = -50.0;

        // Mark roughly every third slot resident.
        let mut resident = CutRowMap::new(cap, 0);
        for slot in (0..cap).step_by(3) {
            resident.insert(slot);
        }

        let current_iteration = 10;
        // Finite k1 window so the recency filter is exercised alongside residency.
        let p = params(7, 1e-9, Some(5));

        let (ref_alpha_bits, ref_selected) = per_row_reference(
            &pool,
            &idx,
            &primal,
            &col_scale,
            &resident,
            &p,
            current_iteration,
        );

        let mut scratch = DcsScoringScratch::default();
        let mut out = Vec::new();
        let count = score_violated_candidates(
            &pool,
            &idx,
            &primal,
            &col_scale,
            &resident,
            &p,
            current_iteration,
            &mut scratch,
            &mut out,
        );

        // The gathered candidate count must match the reference candidate count.
        assert_eq!(
            scratch.alpha.len(),
            ref_alpha_bits.len(),
            "batched and per-row reference must gather the same candidate set"
        );
        // Activities bit-identical, gather-order for gather-order.
        let batched_alpha_bits: Vec<u64> = scratch.alpha.iter().map(|a| a.to_bits()).collect();
        assert_eq!(
            batched_alpha_bits, ref_alpha_bits,
            "batched alpha must be bit-identical to the per-row reference"
        );
        // Selected slot ids AND order bit-identical.
        assert_eq!(
            out, ref_selected,
            "batched out_selected must match the per-row reference exactly"
        );
        assert!(count >= out.len());

        // Witness guard: the fixture must actually exercise the batched path —
        // multiple candidates gathered, at least one violation selected, and the
        // nadic cap engaged — so equality is non-vacuous.
        assert!(
            scratch.alpha.len() > 1,
            "fixture must gather >1 candidate (got {})",
            scratch.alpha.len()
        );
        assert!(
            !out.is_empty(),
            "fixture must select at least one violation"
        );
        assert!(
            count > out.len(),
            "fixture must have more violations ({count}) than the nadic cap ({})",
            out.len()
        );
    }

    /// AC2: a scoring pass issues exactly ONE batched GEMM over all `k` eligible
    /// candidates, not `k` single-row calls. Observed via the gather buffers: all
    /// `k` candidates are in `cand_slots` / `cand_coef_block` (k_rows × n_state)
    /// and `alpha` holds exactly `k` activities after the single call.
    #[test]
    fn batched_scoring_single_gemm_per_pass() {
        let idx = indexer();
        let mut pool = empty_pool();
        // Three active, non-resident, in-window candidates (k = 3).
        add(&mut pool, 0, 5.0, &[1.0, 0.0], 10);
        add(&mut pool, 1, 2.0, &[0.0, 1.0], 10);
        add(&mut pool, 2, 9.0, &[0.5, 0.5], 10);
        let primal = vec![0.0_f64; PRIMAL_LEN];
        let resident = empty_resident();
        let p = params(10, 1e-10, None);
        let mut scratch = DcsScoringScratch::default();
        let mut out = Vec::new();

        let _ = score_violated_candidates(
            &pool,
            &idx,
            &primal,
            &[],
            &resident,
            &p,
            10,
            &mut scratch,
            &mut out,
        );

        let k = 3;
        assert_eq!(scratch.cand_slots.len(), k, "all k candidates gathered");
        assert_eq!(
            scratch.cand_slots,
            vec![0, 1, 2],
            "ascending-slot gather order"
        );
        assert_eq!(
            scratch.cand_coef_block.len(),
            k * N_STATE,
            "coef block is exactly k_rows × n_state — one batched GEMM input"
        );
        assert_eq!(
            scratch.alpha.len(),
            k,
            "alpha holds exactly k activities from the single batched GEMM"
        );
    }

    /// AC2 (filtered): only the eligible (non-resident, in-window) candidates are
    /// gathered into the single batched GEMM; filtered-out slots never enter the
    /// gather buffers.
    #[test]
    fn batched_scoring_gathers_only_eligible() {
        let idx = indexer();
        let mut pool = empty_pool();
        add(&mut pool, 0, 5.0, &[0.0, 0.0], 8); // age 2 — in window
        add(&mut pool, 1, 2.0, &[0.0, 0.0], 2); // age 8 — OUT of window (skipped)
        add(&mut pool, 2, 9.0, &[0.0, 0.0], 9); // age 1 — in window, but resident
        add(&mut pool, 3, 4.0, &[0.0, 0.0], 9); // age 1 — in window, eligible
        let primal = vec![0.0_f64; PRIMAL_LEN];
        let mut resident = empty_resident();
        resident.insert(2); // slot 2 resident
        let p = params(10, 1e-10, Some(5)); // k1 = 5
        let mut scratch = DcsScoringScratch::default();
        let mut out = Vec::new();

        let _ = score_violated_candidates(
            &pool,
            &idx,
            &primal,
            &[],
            &resident,
            &p,
            10,
            &mut scratch,
            &mut out,
        );

        // Only slots 0 and 3 survive both filters; the single GEMM scores them.
        assert_eq!(
            scratch.cand_slots,
            vec![0, 3],
            "out-of-window (1) and resident (2) slots excluded from the gather"
        );
        assert_eq!(scratch.alpha.len(), 2);
    }

    /// AC2 (empty): an empty candidate set produces an empty gather and no
    /// activities — the single GEMM is a no-op (k_rows = 0).
    #[test]
    fn batched_scoring_empty_candidates_no_op() {
        let idx = indexer();
        let pool = empty_pool(); // no cuts at all
        let primal = vec![0.0_f64; PRIMAL_LEN];
        let resident = empty_resident();
        let p = params(10, 1e-10, None);
        let mut scratch = DcsScoringScratch::default();
        let mut out = Vec::new();

        let count = score_violated_candidates(
            &pool,
            &idx,
            &primal,
            &[],
            &resident,
            &p,
            10,
            &mut scratch,
            &mut out,
        );

        assert_eq!(count, 0);
        assert!(scratch.cand_slots.is_empty());
        assert!(scratch.cand_coef_block.is_empty());
        assert!(scratch.alpha.is_empty());
        assert!(out.is_empty());
    }

    /// AC2 (all-resident): every active cut is resident → nothing is gathered and
    /// the single GEMM is a no-op, matching the empty-candidate behavior.
    #[test]
    fn batched_scoring_all_resident_no_candidates() {
        let idx = indexer();
        let mut pool = empty_pool();
        add(&mut pool, 0, 5.0, &[1.0, 0.0], 10);
        add(&mut pool, 1, 2.0, &[0.0, 1.0], 10);
        let primal = vec![0.0_f64; PRIMAL_LEN];
        let mut resident = empty_resident();
        resident.insert(0);
        resident.insert(1);
        let p = params(10, 1e-10, None);
        let mut scratch = DcsScoringScratch::default();
        let mut out = Vec::new();

        let count = score_violated_candidates(
            &pool,
            &idx,
            &primal,
            &[],
            &resident,
            &p,
            10,
            &mut scratch,
            &mut out,
        );

        assert_eq!(count, 0, "all candidates resident → none scored");
        assert!(scratch.cand_slots.is_empty(), "all-resident → empty gather");
        assert!(scratch.alpha.is_empty());
        assert!(out.is_empty());
    }

    /// AC3: the new scratch buffers are growth-only. After a `reserve` sized to
    /// the pool, repeated scoring passes never reallocate — capacities are stable
    /// and the per-pass `clear()` keeps lengths bounded by the eligible count.
    #[test]
    fn batched_scoring_scratch_is_growth_only() {
        let idx = indexer();
        let cap = 32;
        let pool = random_pool(cap, 0x00C0_FFEE_0BAD_CAFE);
        let primal = vec![0.0_f64; PRIMAL_LEN];
        let resident = empty_resident();
        let p = params(10, 1e-10, None);
        let mut scratch = DcsScoringScratch::default();
        let mut out = Vec::new();

        // Pre-reserve to the pool worst case.
        scratch.reserve(N_STATE, cap);
        let coef_cap = scratch.cand_coef_block.capacity();
        let alpha_cap = scratch.alpha.capacity();
        let slots_cap = scratch.cand_slots.capacity();
        let viol_cap = scratch.violations.capacity();
        let state_cap = scratch.unscaled_state.capacity();
        assert!(
            coef_cap >= cap * N_STATE,
            "coef block reserved to cap*n_state"
        );
        assert!(alpha_cap >= cap);
        assert!(slots_cap >= cap);

        // Several passes must not grow any capacity (no steady-state allocation).
        for _ in 0..5 {
            let _ = score_violated_candidates(
                &pool,
                &idx,
                &primal,
                &[],
                &resident,
                &p,
                10,
                &mut scratch,
                &mut out,
            );
            assert_eq!(scratch.cand_coef_block.capacity(), coef_cap);
            assert_eq!(scratch.alpha.capacity(), alpha_cap);
            assert_eq!(scratch.cand_slots.capacity(), slots_cap);
            assert_eq!(scratch.violations.capacity(), viol_cap);
            assert_eq!(scratch.unscaled_state.capacity(), state_cap);
        }
    }

    // -----------------------------------------------------------------------
    // lazy_solve fixtures
    // -----------------------------------------------------------------------

    // Synthetic LP for StageIndexer::new(1, 0): n_state = 1, theta = col 3.
    //   Columns: 0 = state x0 (pinned to 2.0), 1 and 2 fixed to 0, 3 = theta.
    //   No structural rows (num_rows = 0) — cuts are the only rows.
    //   Objective: minimize theta. So at the optimum theta equals the max cut
    //   floor `intercept + coeff*x0` over the resident cuts.
    const STATE_X0: f64 = 2.0;
    const LAZY_THETA_COL: usize = 3;

    fn lazy_indexer() -> StageIndexer {
        StageIndexer::new(1, 0)
    }

    /// Cut-free core template with x0 pinned to `STATE_X0` and theta free.
    fn core_template() -> StageTemplate {
        StageTemplate {
            num_cols: 4,
            num_rows: 0,
            num_nz: 0,
            col_starts: vec![0_i32; 5], // num_cols + 1, all zero (no structural nz)
            row_indices: Vec::new(),
            values: Vec::new(),
            // x0 pinned to STATE_X0; cols 1,2 fixed to 0; theta free (bounded box).
            col_lower: vec![STATE_X0, 0.0, 0.0, -1.0e6],
            col_upper: vec![STATE_X0, 0.0, 0.0, 1.0e6],
            objective: vec![0.0, 0.0, 0.0, 1.0], // minimize theta
            row_lower: Vec::new(),
            row_upper: Vec::new(),
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Three cuts (state_dim = 1), floors at x0 = 2.0:
    ///   slot 0: intercept 1, coeff [0]   → floor 1.0
    ///   slot 1: intercept 0, coeff [1]   → floor 2.0
    ///   slot 2: intercept 5, coeff [0]   → floor 5.0  (the binding cut)
    /// All-cuts optimum: theta = 5.0, objective = 5.0; binding slot = 2.
    fn make_three_cut_pool() -> CutPool {
        let mut pool = CutPool::new(16, 1, 16, 0);
        pool.add_cut(0, 0, 1.0, &[0.0]);
        pool.add_cut(0, 1, 0.0, &[1.0]);
        pool.add_cut(0, 2, 5.0, &[0.0]);
        for slot in 0..3 {
            pool.metadata[slot].iteration_generated = 1;
        }
        pool
    }

    /// Four cuts (state_dim = 1) whose binding cut depends on x0:
    ///   slot 0: intercept 1, coeff [0]   → floor 1
    ///   slot 1: intercept 0, coeff [1]   → floor x0
    ///   slot 2: intercept 5, coeff [0]   → floor 5
    ///   slot 3: intercept 0, coeff [2]   → floor 2·x0
    /// At x0 = 2 the binding cut is slot 2 (floor 5); at x0 = 10 it is slot 3
    /// (floor 20). Used to exercise the opening-reuse continue path.
    fn make_four_cut_pool() -> CutPool {
        let mut pool = CutPool::new(16, 1, 16, 0);
        pool.add_cut(0, 0, 1.0, &[0.0]);
        pool.add_cut(0, 1, 0.0, &[1.0]);
        pool.add_cut(0, 2, 5.0, &[0.0]);
        pool.add_cut(0, 3, 0.0, &[2.0]);
        for slot in 0..4 {
            pool.metadata[slot].iteration_generated = 1;
        }
        pool
    }

    fn active_profiled() -> ProfiledSolver<ActiveSolver> {
        ProfiledSolver::new(ActiveSolver::new().expect("ActiveSolver::new()"))
    }

    fn ctx() -> DcsSolveContext {
        DcsSolveContext {
            stage_index: 0,
            scenario_index: 0,
            iteration: Some(10),
            continue_carry: false,
        }
    }

    fn lazy_params(nadic: u32, max_inner: u32) -> DcsParams {
        DcsParams {
            k1: None,
            nadic,
            max_inner_iterations: max_inner,
            ..DcsParams::default()
        }
    }

    /// Reference all-cuts optimum: load core, append every active slot, solve.
    fn solve_all_cuts(pool: &CutPool, indexer: &StageIndexer) -> (f64, f64) {
        let mut solver = active_profiled();
        let core = core_template();
        solver.load_model(&core);
        let mut row_map = CutRowMap::new(pool.populated_count, core.num_rows);
        let mut batch = RowBatch {
            num_rows: 0,
            row_starts: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            row_lower: Vec::new(),
            row_upper: Vec::new(),
        };
        let all_slots: Vec<u32> = (0..pool.populated_count as u32).collect();
        crate::forward::append_slots_to_lp(
            &mut solver,
            pool,
            &all_slots,
            indexer,
            &[],
            &mut row_map,
            &mut batch,
        );
        let view = solver.solve(None).expect("all-cuts solve must succeed");
        (view.objective, view.primal[LAZY_THETA_COL])
    }

    // -----------------------------------------------------------------------
    // lazy_solve tests (real solver)
    // -----------------------------------------------------------------------

    /// Headline exactness gate: lazy_solve with an initial subset that omits the
    /// binding cut converges to the all-cuts optimum.
    #[test]
    fn lazy_solve_exact_matches_all_cuts() {
        let indexer = lazy_indexer();
        let pool = make_three_cut_pool();
        let (all_obj, all_theta) = solve_all_cuts(&pool, &indexer);

        let mut solver = active_profiled();
        let core = core_template();
        let mut scratch = DcsSolveScratch::default();
        let params = lazy_params(10, 50);

        // Caller owns the model load (and, in production, any bounds patch).
        solver.load_model(&core);
        lazy_solve_preloaded(
            &mut solver,
            &core,
            &pool,
            &indexer,
            &[],
            None,
            &[0, 1], // omit binding slot 2
            &params,
            &mut scratch,
            ctx(),
        )
        .expect("lazy_solve_preloaded must succeed");
        let view = scratch.result_view();

        assert!(
            (view.objective - all_obj).abs() < 1e-9,
            "objective {} != all-cuts {all_obj}",
            view.objective
        );
        assert!(
            (view.primal[LAZY_THETA_COL] - all_theta).abs() < 1e-9,
            "theta {} != all-cuts {all_theta}",
            view.primal[LAZY_THETA_COL]
        );
        // Bit-identical with trivial epsilon/scale.
        assert_eq!(view.objective, all_obj);
        assert_eq!(view.primal[LAZY_THETA_COL], all_theta);
    }

    /// With an initial subset that already contains the binding cut, the loop
    /// adds no further rows and returns after the first solve.
    ///
    /// AC1: the no-violation terminating path issues EXACTLY ONE LP solve (the
    /// initial solve), not two — the redundant exit re-solve is eliminated. The
    /// resident subset already reproduces the all-cuts optimum, so the live view
    /// is copied into the result buffers and returned with no re-solve.
    #[test]
    fn lazy_solve_no_violation_stops_immediately() {
        let indexer = lazy_indexer();
        let pool = make_three_cut_pool();
        let mut solver = active_profiled();
        let core = core_template();
        let mut scratch = DcsSolveScratch::default();
        let params = lazy_params(10, 50);

        // Initial subset includes the binding slot 2.
        solver.load_model(&core);
        let solves_before = solver.statistics().solve_count;
        lazy_solve_preloaded(
            &mut solver,
            &core,
            &pool,
            &indexer,
            &[],
            None,
            &[0, 1, 2],
            &params,
            &mut scratch,
            ctx(),
        )
        .expect("lazy_solve_preloaded must succeed");
        let solve_delta = solver.statistics().solve_count - solves_before;
        let view = scratch.result_view();

        // The optimum is the binding cut floor; no growth needed.
        assert_eq!(view.primal[LAZY_THETA_COL], 5.0);
        // out_selected from the (single) scoring pass that found no violation.
        assert!(scratch.out_selected.is_empty());
        // AC1: exactly ONE solve (was 2 before the redundant exit re-solve was
        // removed): the initial solve, then the no-violation copy-and-return.
        assert_eq!(
            solve_delta, 1,
            "no-violation path must issue exactly 1 solve (no redundant exit re-solve)"
        );
    }

    /// Omitting the binding cut forces at least one inner add_rows, and the
    /// final theta reflects the now-resident binding cut.
    ///
    /// AC2: the one-addition path issues EXACTLY TWO LP solves (was 3: initial +
    /// add/resolve + redundant exit) — the initial solve, then the add-and-resolve
    /// whose rescore finds no violation and copies-and-returns with no third solve.
    #[test]
    fn lazy_solve_grows_to_include_binding_cut() {
        let indexer = lazy_indexer();
        let pool = make_three_cut_pool();
        let mut solver = active_profiled();
        let core = core_template();
        let mut scratch = DcsSolveScratch::default();
        let params = lazy_params(10, 50);

        solver.load_model(&core);
        let solves_before = solver.statistics().solve_count;
        lazy_solve_preloaded(
            &mut solver,
            &core,
            &pool,
            &indexer,
            &[],
            None,
            &[0, 1], // omit binding slot 2
            &params,
            &mut scratch,
            ctx(),
        )
        .expect("lazy_solve_preloaded must succeed");
        let solve_delta = solver.statistics().solve_count - solves_before;
        let view = scratch.result_view();

        assert_eq!(
            view.primal[LAZY_THETA_COL], 5.0,
            "final theta must reflect the added binding cut"
        );
        // AC2: exactly TWO solves (initial + one add/resolve), no redundant exit.
        assert_eq!(
            solve_delta, 2,
            "one-addition path must issue exactly 2 solves (no redundant exit re-solve)"
        );
    }

    /// Opening-reuse continue path: load the cut-free core ONCE, solve a first
    /// "opening" fresh at x0 = 2, then re-pin x0 = 10 and solve a second opening
    /// by CONTINUING (`continue_carry = true`) — reusing the loaded LP, the
    /// carried resident cut rows, and the warm basis from the first solve. Each
    /// opening must converge to the all-cuts optimum for its own pinned state,
    /// and the second must add the cut (slot 3) that binds only at x0 = 10 on top
    /// of the carried residents {0,1,2}. This is the now-mandatory backward
    /// opening-reuse mechanism (ReuseContinue) exercised in isolation.
    ///
    /// AC3: a third opening C continues at x0 = 2 where the binding cut (slot 2)
    /// is ALREADY resident from the carried set, so the lazy loop adds nothing and
    /// must issue EXACTLY ONE solve on the continue-carry no-add path.
    #[test]
    fn lazy_solve_continue_carry_exact_across_openings() {
        let indexer = lazy_indexer();
        let pool = make_four_cut_pool();
        let core = core_template();
        let params = lazy_params(10, 50);
        let mut solver = active_profiled();
        // ONE scratch + ONE loaded model shared across the openings: the row map
        // and warm basis must persist across the continue calls.
        let mut scratch = DcsSolveScratch::default();
        solver.load_model(&core);

        // Opening A (x0 = 2, fresh): seed omits the binding slot 2; the lazy loop
        // discovers and adds it. All-cuts optimum at x0 = 2 is theta = 5.
        lazy_solve_preloaded(
            &mut solver,
            &core,
            &pool,
            &indexer,
            &[],
            None,
            &[0, 1],
            &params,
            &mut scratch,
            DcsSolveContext {
                continue_carry: false,
                ..ctx()
            },
        )
        .expect("opening A (fresh) must solve");
        assert_eq!(
            scratch.result_view().primal[LAZY_THETA_COL],
            5.0,
            "opening A theta (x0=2)"
        );

        // Re-pin the incoming state to x0 = 10 (what `patch_opening_bounds` does
        // per opening) — only the bounds change; the loaded LP and the carried
        // resident cut rows persist.
        solver.set_col_bounds(&[0], &[10.0], &[10.0]);

        // Opening B (x0 = 10, continue): no reload, no re-seed; warm-carry A's LP
        // and basis. The binding cut at x0 = 10 is slot 3 (floor 20), which is NOT
        // resident after A, so the continue loop must add it. All-cuts optimum at
        // x0 = 10 is theta = 20.
        lazy_solve_preloaded(
            &mut solver,
            &core,
            &pool,
            &indexer,
            &[],
            None,
            &[], // ignored in continue mode
            &params,
            &mut scratch,
            DcsSolveContext {
                continue_carry: true,
                ..ctx()
            },
        )
        .expect("opening B (continue) must solve");
        assert_eq!(
            scratch.result_view().primal[LAZY_THETA_COL],
            20.0,
            "opening B theta (x0=10) must reach the all-cuts optimum via the \
             carried-LP continue path (slot 3 added on top of the carried set)"
        );

        // Opening C (x0 = 2 again, continue, NO add): re-pin back to x0 = 2 where
        // the binding cut (slot 2, floor 5) is already resident from A. The
        // continue loop finds no violation and must issue EXACTLY ONE solve — the
        // continue-carry no-add path (AC3).
        solver.set_col_bounds(&[0], &[2.0], &[2.0]);
        let solves_before = solver.statistics().solve_count;
        lazy_solve_preloaded(
            &mut solver,
            &core,
            &pool,
            &indexer,
            &[],
            None,
            &[], // ignored in continue mode
            &params,
            &mut scratch,
            DcsSolveContext {
                continue_carry: true,
                ..ctx()
            },
        )
        .expect("opening C (continue, no add) must solve");
        let solve_delta = solver.statistics().solve_count - solves_before;
        assert_eq!(
            scratch.result_view().primal[LAZY_THETA_COL],
            5.0,
            "opening C theta (x0=2) reverts to the carried binding cut (slot 2)"
        );
        // AC3: continue-carry no-add path issues exactly ONE solve.
        assert_eq!(
            solve_delta, 1,
            "continue-carry no-add opening must issue exactly 1 solve (no redundant exit re-solve)"
        );
    }

    // -----------------------------------------------------------------------
    // scoring-time instrumentation tests
    // -----------------------------------------------------------------------

    /// A freshly-constructed scratch starts with a zero scoring-time accumulator.
    #[test]
    fn scoring_time_default_is_zero() {
        let scratch = DcsSolveScratch::default();
        assert_eq!(scratch.scoring_time_seconds, 0.0);
    }

    /// A solve whose seed omits the binding cut runs the inner loop (>= 2 scoring
    /// passes) and strictly increases the cumulative scoring accumulator.
    #[test]
    fn scoring_time_increases_when_inner_loop_runs() {
        let indexer = lazy_indexer();
        let pool = make_three_cut_pool();
        let mut solver = active_profiled();
        let core = core_template();
        let mut scratch = DcsSolveScratch::default();
        let params = lazy_params(10, 50);

        let before = scratch.scoring_time_seconds;
        solver.load_model(&core);
        lazy_solve_preloaded(
            &mut solver,
            &core,
            &pool,
            &indexer,
            &[],
            None,
            &[0, 1], // omit binding slot 2 → inner loop runs
            &params,
            &mut scratch,
            ctx(),
        )
        .expect("lazy_solve_preloaded must succeed");

        assert!(
            scratch.scoring_time_seconds > before,
            "scoring accumulator {} must exceed its pre-call value {before}",
            scratch.scoring_time_seconds
        );
    }

    /// The rows-in-LP accumulators track the resident cut-row count per solve:
    /// they start at zero, fold one term per solve (cumulative, never reset
    /// between solves on the same scratch), and `max` tracks the peak. Seeding
    /// all three cuts resident gives a resident set of 3 with no growth, so two
    /// fresh solves on the same scratch yield count == 2, sum == 6, max == 3.
    #[test]
    fn rows_in_lp_tracks_resident_set_size_per_solve() {
        let indexer = lazy_indexer();
        let pool = make_three_cut_pool();
        let mut solver = active_profiled();
        let core = core_template();
        let mut scratch = DcsSolveScratch::default();
        let params = lazy_params(10, 50);

        // Defaults are zero before any solve.
        assert_eq!(scratch.rows_in_lp_sum, 0);
        assert_eq!(scratch.rows_in_lp_count, 0);
        assert_eq!(scratch.rows_in_lp_max, 0);

        for _ in 0..2 {
            solver.load_model(&core);
            lazy_solve_preloaded(
                &mut solver,
                &core,
                &pool,
                &indexer,
                &[],
                None,
                &[0, 1, 2], // all three cuts resident from the seed → no growth
                &params,
                &mut scratch,
                ctx(),
            )
            .expect("lazy_solve_preloaded must succeed");
        }

        assert_eq!(scratch.rows_in_lp_count, 2, "one term folded per solve");
        assert_eq!(
            scratch.rows_in_lp_sum, 6,
            "3 resident cut rows per solve, summed over 2 solves"
        );
        assert_eq!(
            scratch.rows_in_lp_max, 3,
            "peak resident set is the 3 seeded cuts"
        );
    }

    /// A solve whose seed already contains the binding cut performs exactly one
    /// scoring pass (no growth): the accumulator increases by that single pass
    /// and the objective/theta result is unchanged from the pre-instrumentation
    /// behavior (still the binding-cut optimum).
    #[test]
    fn scoring_time_single_pass_result_unchanged() {
        let indexer = lazy_indexer();
        let pool = make_three_cut_pool();
        let mut solver = active_profiled();
        let core = core_template();
        let mut scratch = DcsSolveScratch::default();
        let params = lazy_params(10, 50);

        let before = scratch.scoring_time_seconds;
        solver.load_model(&core);
        lazy_solve_preloaded(
            &mut solver,
            &core,
            &pool,
            &indexer,
            &[],
            None,
            &[0, 1, 2], // binding cut already resident → single scoring pass, no growth
            &params,
            &mut scratch,
            ctx(),
        )
        .expect("lazy_solve_preloaded must succeed");
        let view = scratch.result_view();

        // Single scoring pass still advances the accumulator.
        assert!(
            scratch.scoring_time_seconds >= before,
            "scoring accumulator must not decrease"
        );
        // Result identical to the pre-instrumentation single-pass behavior: the
        // binding-cut floor, no inner growth.
        assert_eq!(
            view.primal[LAZY_THETA_COL], 5.0,
            "single-pass optimum must be the binding-cut floor"
        );
        assert_eq!(view.objective, 5.0);
        assert!(
            scratch.out_selected.is_empty(),
            "no violation → no slots selected for growth"
        );
    }

    /// max_inner_iterations = 1 forces the TC fallback after one inner add; it
    /// must terminate and still return the all-cuts-equivalent optimum.
    ///
    /// AC5: the cap-hit fallback scores the live primal, appends remaining
    /// violated slots, and solves once to return — preserving the all-cuts
    /// optimum. Solve count is unchanged from before this ticket on the TC path:
    /// initial + one capped add/resolve + the final TC solve = 3.
    #[test]
    fn lazy_solve_tc_fallback_terminates() {
        let indexer = lazy_indexer();
        let pool = make_three_cut_pool();
        let (all_obj, all_theta) = solve_all_cuts(&pool, &indexer);

        let mut solver = active_profiled();
        let core = core_template();
        let mut scratch = DcsSolveScratch::default();
        // nadic = 1 + cap = 1: the loop can add at most one slot before the cap
        // fires, then the TC fallback adds the rest.
        let params = lazy_params(1, 1);

        solver.load_model(&core);
        let solves_before = solver.statistics().solve_count;
        lazy_solve_preloaded(
            &mut solver,
            &core,
            &pool,
            &indexer,
            &[],
            None,
            &[], // omit ALL cuts to force ≥2 inner iterations without the cap
            &params,
            &mut scratch,
            ctx(),
        )
        .expect("lazy_solve_preloaded TC fallback must succeed");
        let solve_delta = solver.statistics().solve_count - solves_before;
        let view = scratch.result_view();

        assert!((view.objective - all_obj).abs() < 1e-9);
        assert!((view.primal[LAZY_THETA_COL] - all_theta).abs() < 1e-9);
        // AC5: TC fallback solve count is preserved (initial + capped add + final).
        assert_eq!(
            solve_delta, 3,
            "TC fallback issues initial + one capped add/resolve + final TC solve = 3"
        );
    }

    /// Determinism: identical inputs → identical objective across two runs.
    #[test]
    fn lazy_solve_is_deterministic() {
        let indexer = lazy_indexer();
        let pool = make_three_cut_pool();
        let core = core_template();
        let params = lazy_params(10, 50);

        let run = || {
            let mut solver = active_profiled();
            let mut scratch = DcsSolveScratch::default();
            solver.load_model(&core);
            lazy_solve_preloaded(
                &mut solver,
                &core,
                &pool,
                &indexer,
                &[],
                None,
                &[0],
                &params,
                &mut scratch,
                ctx(),
            )
            .expect("lazy_solve_preloaded must succeed");
            scratch.result_view().objective
        };

        assert_eq!(run(), run(), "objective must be deterministic");
    }

    /// AC6: the reused `res_*` result buffers are growth-only. Once warmed to the
    /// LP's solution size, repeated `lazy_solve_preloaded` calls refill them via
    /// `clear()` + `extend_from_slice` without reallocating — capacities are
    /// stable (no steady-state heap allocation on the result-copy path). Also
    /// pins that `result_view()` round-trips the solution faithfully.
    ///
    /// Each call reloads the cut-free core first, exactly as the production
    /// forward/backward/sim paths do per (stage, solve): without the reload the
    /// solver model would accumulate cut rows across calls and the dual length
    /// (and thus `res_dual`) would grow — a fixture artifact, not a real leak.
    #[test]
    fn lazy_solve_result_buffers_growth_only() {
        let indexer = lazy_indexer();
        let pool = make_three_cut_pool();
        let mut solver = active_profiled();
        let core = core_template();
        let mut scratch = DcsSolveScratch::default();
        let params = lazy_params(10, 50);

        let call = |solver: &mut ProfiledSolver<ActiveSolver>, scratch: &mut DcsSolveScratch| {
            // Reload the cut-free core so each call starts from a clean row count
            // (mirrors the caller's per-(stage, solve) `load_model`).
            solver.load_model(&core);
            lazy_solve_preloaded(
                solver,
                &core,
                &pool,
                &indexer,
                &[],
                None,
                &[0, 1], // omit binding slot 2 → at least one add, fills res_* twice
                &params,
                scratch,
                ctx(),
            )
            .expect("lazy_solve_preloaded must succeed");
        };

        // Two warm-up calls so the result buffers reach their steady-state
        // capacity before we start asserting stability.
        call(&mut solver, &mut scratch);
        call(&mut solver, &mut scratch);
        let primal_cap = scratch.res_primal.capacity();
        let dual_cap = scratch.res_dual.capacity();
        let rc_cap = scratch.res_reduced_costs.capacity();
        assert!(
            !scratch.res_primal.is_empty(),
            "result primal must be filled"
        );
        assert_eq!(
            scratch.result_view().primal[LAZY_THETA_COL],
            5.0,
            "warmed result must be the binding optimum"
        );

        // Several more calls must not grow any result-buffer capacity.
        for _ in 0..3 {
            call(&mut solver, &mut scratch);
            assert_eq!(
                scratch.res_primal.capacity(),
                primal_cap,
                "res_primal capacity must be stable (growth-only)"
            );
            assert_eq!(
                scratch.res_dual.capacity(),
                dual_cap,
                "res_dual capacity must be stable (growth-only)"
            );
            assert_eq!(
                scratch.res_reduced_costs.capacity(),
                rc_cap,
                "res_reduced_costs capacity must be stable (growth-only)"
            );
            // result_view() still reflects the same optimum each call.
            assert_eq!(scratch.result_view().primal[LAZY_THETA_COL], 5.0);
        }
    }

    // -----------------------------------------------------------------------
    // lazy_solve test: cold mid-loop re-solve tolerance (mock solver)
    // -----------------------------------------------------------------------

    /// A mock solver that returns one primal vector on the first solve and a
    /// different one on every subsequent solve, never reporting a usable warm
    /// basis (it ignores the passed basis entirely — simulating an escalation
    /// that cold-started). Both solves are valid (non-error).
    struct TwoPhaseMock {
        first: Vec<f64>,
        rest: Vec<f64>,
        call_count: usize,
        buf: Vec<f64>,
        empty: Vec<f64>,
    }

    impl TwoPhaseMock {
        fn new(first: Vec<f64>, rest: Vec<f64>) -> Self {
            Self {
                first,
                rest,
                call_count: 0,
                buf: Vec::new(),
                empty: Vec::new(),
            }
        }
    }

    impl SolverInterface for TwoPhaseMock {
        type Profile = ActiveProfile;
        fn apply_profile(&mut self, _profile: &ActiveProfile) {}
        fn solver_name_version(&self) -> String {
            "TwoPhaseMock 0.0.0".to_string()
        }
        fn load_model(&mut self, _template: &StageTemplate) {}
        fn add_rows(&mut self, _rows: &RowBatch) {}
        fn set_row_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
        fn set_col_bounds(&mut self, _i: &[usize], _l: &[f64], _u: &[f64]) {}
        fn solve(
            &mut self,
            _basis: Option<&Basis>,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            // Always behave as a cold solve: ignore any passed basis.
            let src = if self.call_count == 0 {
                &self.first
            } else {
                &self.rest
            };
            self.call_count += 1;
            self.buf.clone_from(src);
            Ok(cobre_solver::SolutionView {
                objective: self.buf[LAZY_THETA_COL],
                primal: &self.buf,
                dual: &self.empty,
                reduced_costs: &self.empty,
                iterations: 0,
                solve_time_seconds: 0.0,
            })
        }
        fn get_basis(&mut self, _out: &mut Basis) {}
        fn statistics(&self) -> SolverStatistics {
            SolverStatistics {
                solve_count: self.call_count as u64,
                ..SolverStatistics::default()
            }
        }
        fn statistics_into(&self, out: &mut SolverStatistics) {
            out.copy_from(&self.statistics());
        }
        fn name(&self) -> &'static str {
            "TwoPhaseMock"
        }
    }

    /// A mid-loop `solve(None)` that discarded its warm basis (the mock ignores
    /// the basis entirely) must be tolerated: lazy_solve completes and reaches
    /// the no-violation stop.
    #[test]
    fn lazy_solve_tolerates_cold_mid_loop_resolve() {
        let indexer = lazy_indexer();
        // One cut: intercept 5, coeff [0] → floor 5.0 at x0 = 2.
        let mut pool = CutPool::new(16, 1, 16, 0);
        pool.add_cut(0, 0, 5.0, &[0.0]);
        pool.metadata[0].iteration_generated = 1;

        // Primal layout: [x0, c1, c2, theta]. First solve reports theta = 0
        // (cut 0 floor 5 > 0 → violated). After adding cut 0, the mock reports
        // theta = 5 (floor 5 == theta → not violated → stop).
        let first = vec![STATE_X0, 0.0, 0.0, 0.0];
        let rest = vec![STATE_X0, 0.0, 0.0, 5.0];
        let mut solver = ProfiledSolver::new(TwoPhaseMock::new(first, rest));
        let core = core_template();
        let mut scratch = DcsSolveScratch::default();
        let params = lazy_params(10, 50);

        solver.load_model(&core);
        lazy_solve_preloaded(
            &mut solver,
            &core,
            &pool,
            &indexer,
            &[],
            None,
            &[], // omit cut 0 so the loop must add it mid-loop
            &params,
            &mut scratch,
            ctx(),
        )
        .expect("lazy_solve_preloaded must tolerate a cold mid-loop re-solve");
        let view = scratch.result_view();

        assert_eq!(
            view.primal[LAZY_THETA_COL], 5.0,
            "loop reaches the no-violation stop on the second (cold) solve"
        );
    }

    // -----------------------------------------------------------------------
    // build_initial_resident_set fixtures
    // -----------------------------------------------------------------------

    /// Build a pool of `n` populated, state_dim-1 cuts with explicit per-slot
    /// `(active, iteration_generated, last_active_iter)` metadata. Local copy of
    /// the `cut_selection` test-helper pattern (kept private to this module).
    #[allow(clippy::cast_possible_truncation)]
    fn seed_pool(specs: &[(bool, u64, u64)]) -> CutPool {
        let n = specs.len();
        let mut pool = CutPool::new(n.max(1), 1, n.max(1) as u32, 0);
        for (i, &(active, iteration_generated, last_active_iter)) in specs.iter().enumerate() {
            pool.add_cut(0, i as u32, 0.0, &[0.0]);
            pool.metadata[i] = CutMetadata {
                iteration_generated,
                forward_pass_index: i as u32,
                active_count: 0,
                last_active_iter,
            };
            pool.active[i] = active;
        }
        pool.cached_active_count = specs.iter().filter(|&&(a, _, _)| a).count();
        pool
    }

    // -----------------------------------------------------------------------
    // build_initial_resident_set tests
    // -----------------------------------------------------------------------

    /// AC1: slots 0..4 with last_active_iter = [10, 8, 3, 10, 6], all active,
    /// all generated at iter 1, current = 10, k2 = 5. Slot 2 excluded
    /// (10 - 3 = 7 > 5); the rest are within the window.
    #[test]
    fn seeds_within_k2_window() {
        let pool = seed_pool(&[
            (true, 1, 10),
            (true, 1, 8),
            (true, 1, 3),
            (true, 1, 10),
            (true, 1, 6),
        ]);
        let mut out = Vec::new();
        build_initial_resident_set(&pool, 10, 5, &mut out);
        assert_eq!(out, vec![0, 1, 3, 4]);
    }

    /// AC2: a current-iteration cut (iteration_generated == current) is always
    /// seeded even when its last_active_iter is far in the past.
    #[test]
    fn always_seeds_current_iteration_cuts() {
        // Slot 0: out of window (10 - 1 = 9 > 2) but generated this iteration.
        // Slot 1: in window.
        let pool = seed_pool(&[(true, 10, 1), (true, 1, 9)]);
        let mut out = Vec::new();
        build_initial_resident_set(&pool, 10, 2, &mut out);
        assert_eq!(
            out,
            vec![0, 1],
            "current-iteration slot 0 seeded despite stale last_active_iter"
        );
    }

    /// AC3: an inactive slot within the k2 window is never included.
    #[test]
    fn excludes_inactive_slots() {
        // Slot 0 active in window; slot 1 inactive in window; slot 2 active.
        let pool = seed_pool(&[(true, 1, 10), (false, 1, 10), (true, 1, 10)]);
        let mut out = Vec::new();
        build_initial_resident_set(&pool, 10, 5, &mut out);
        assert_eq!(out, vec![0, 2], "inactive slot 1 excluded");
    }

    /// AC4: the result is strictly ascending and deterministic across repeated
    /// calls on the same metadata (no dependence on iteration order).
    #[test]
    fn result_is_ascending_and_deterministic() {
        let pool = seed_pool(&[
            (true, 1, 10),
            (true, 1, 9),
            (false, 1, 10),
            (true, 1, 8),
            (true, 1, 10),
        ]);
        let mut a = Vec::new();
        let mut b = Vec::new();
        build_initial_resident_set(&pool, 10, 5, &mut a);
        build_initial_resident_set(&pool, 10, 5, &mut b);
        assert_eq!(a, b, "two calls on identical metadata must match");
        assert!(
            a.windows(2).all(|w| w[0] < w[1]),
            "result must be strictly ascending, got {a:?}"
        );
        assert_eq!(a, vec![0, 1, 3, 4]);
    }

    /// AC3 (ticket-017a): the seed keys on **binding recency** via
    /// `last_active_iter`, NOT on generation iteration. A cut generated long ago
    /// (`iteration_generated` well outside the k2 window) but binding recently
    /// (`last_active_iter == i`) must be seeded at iteration `i + d` for
    /// `0 < d <= k2` — the §3.1 clause-1 behavior the binding-count maintenance
    /// restores. The companion "generated but never re-bound" cut, whose
    /// `last_active_iter` stayed at its old generation iteration, is correctly
    /// excluded once it falls outside the window.
    #[test]
    fn seeds_old_generation_cut_that_bound_recently() {
        // current = i + d = 12, k2 = 3 → window covers last_active_iter >= 9.
        //   slot 0: generated at iter 1 (age 11 ≫ k2), but bound at iter 11
        //           → 12 - 11 = 1 <= 3 → SEEDED (recently binding, old gen).
        //   slot 1: generated at iter 1, never re-bound (last_active 1)
        //           → 12 - 1 = 11 > 3, not current-iter → EXCLUDED.
        //   slot 2: generated at iter 1, bound at iter 9 (window edge)
        //           → 12 - 9 = 3 <= 3 → SEEDED.
        let pool = seed_pool(&[(true, 1, 11), (true, 1, 1), (true, 1, 9)]);
        let mut out = Vec::new();
        build_initial_resident_set(&pool, 12, 3, &mut out);
        assert_eq!(
            out,
            vec![0, 2],
            "old-generation cuts that bound within the last k2 iterations are \
             seeded by binding recency; the never-re-bound cut is excluded"
        );
    }

    /// AC: k2 = 0 boundary. Only slots active in the current iteration
    /// (last_active_iter >= current) plus current-iteration cuts are included.
    #[test]
    fn k2_zero_window_boundary() {
        // current = 10, k2 = 0:
        //   slot 0: last_active_iter 10 → 10 - 10 = 0 <= 0 → included.
        //   slot 1: last_active_iter 9  → 10 - 9  = 1 >  0 → excluded (not current-iter).
        //   slot 2: last_active_iter 11 → saturating_sub → 0 <= 0 → included.
        //   slot 3: generated this iteration (last_active_iter 2 stale) → included.
        let pool = seed_pool(&[(true, 1, 10), (true, 1, 9), (true, 1, 11), (true, 10, 2)]);
        let mut out = Vec::new();
        build_initial_resident_set(&pool, 10, 0, &mut out);
        assert_eq!(out, vec![0, 2, 3]);
    }
}
