//! Stage LP patch buffer and stage template builder for SDDP subproblems.
//!
//! This module provides two public facilities:
//!
//! - [`PatchBuffer`]: pre-allocates the parallel arrays consumed by
//!   `SolverInterface::set_row_bounds` and `SolverInterface::set_col_bounds`
//!   and fills them with scenario-dependent values before each LP solve.
//!   Allocating once at training start and reusing the same buffer across all
//!   iterations and stages is critical for hot-path performance: the training
//!   loop calls `fill_forward_patches`, `fill_load_patches`, or
//!   `fill_col_state_patches` millions of times.
//!
//! - [`build_stage_templates`]: constructs a `Vec<StageTemplate>` from a
//!   loaded `System` — one template per study stage.  The template encodes
//!   the full structural LP in CSC format, column/row bounds, and objective
//!   coefficients.  It is built once at startup and shared read-only across
//!   all threads.
//!
//! ## LP layout (Solver Abstraction SS2)
//!
//! The column and row geometry — the contiguous regions, their ordering, and
//! the offset arithmetic — is owned per stage by `StageLayout`, with the
//! state-vector region on [`crate::indexer::StateLayout`] and the non-state
//! study shape on [`crate::indexer::StudyDimensions`].  This module does not
//! restate that geometry; it documents only the per-solve patch sequence layered
//! on top of it (below).
//!
//! The AR dynamics (noise patch target) rows are the water balance constraints
//! beginning at row `base_rows[stage]` (= `N` for a stage with N hydros). The
//! `base_rows` value returned alongside the templates encodes this offset for
//! each stage so that [`PatchBuffer`] can update the correct RHS during
//! forward-pass solves.
//!
//! ### State pinning
//!
//! State pinning lives on **incoming-state columns** rather than rows.
//! Use [`crate::indexer::StateLayout::state_to_lp_incoming_column`] to
//! resolve the column index for state vector entry `j`.  Both forward-pass
//! state pinning (`set_col_bounds(&[col], &[v], &[v])`) and backward-pass
//! cut-subgradient extraction (`view.reduced_costs[col] * col_scale[col]`)
//! use the same column index.  The LP has no state-fixing row range.
//!
//! ## Patch sequence (Training Loop SS4.2a)
//!
//! Each forward-pass solve writes up to `2N + M*K` row patches and up to
//! `N*(1+L) + A*K_max` column-bound patches (where M is the number of
//! stochastic load buses and K is the block count):
//!
//! - **Row buffer** (written by `fill_forward_patches`, `fill_load_patches`,
//!   `fill_z_inflow_patches`):
//!
//! ```text
//! Category 3 -- noise innovation   N rows at base_rows[stage]
//!     patch water_balance_row(base_row, h) = noise[h]   for h in [0, N)
//!
//! Category 4 -- load balance patches   M*K rows (optional, stochastic load)
//!     patch load_row_start + bus_pos[i]*K + blk = load_rhs[i*K + blk]
//!     for i in [0, M), blk in [0, K)
//!
//! Category 5 -- z-inflow RHS      N rows at z_inflow_row_start (= 0)
//!     patch z_inflow_row(h) = base_h + noise_scale_h * eta_h   for h in [0, N)
//! ```
//!
//! - **Column buffer** (written by `fill_col_state_patches`):
//!
//! ```text
//! Category 1 -- incoming storage  cols [storage_in.start, storage_in.end)
//!     col_lower[h] == col_upper[h] == state[h]   for h in [0, N)
//!
//! Category 2 -- AR lag state      cols [inflow_lags.start, inflow_lags.end)
//!     col_lower[N + l*N + h] == col_upper[...] == state[N + l*N + h]
//!     for h in [0, N), l in [0, L)
//!
//! Category 6 -- anticipated state cols [anticipated_state.start, .end)
//!     col_lower[slot] == col_upper[slot] == state[N*(1+L) + slot]
//!     for slot in [0, A*K_max)
//! ```
//!
//! Each column-buffer entry `c` has `col_lower[c] == col_upper[c]` to pin the
//! variable to the state value via tight bounds.  The backward pass writes only
//! the column buffer (state pinning); noise innovations come from the fixed
//! opening tree and are written to the row buffer via `fill_forward_patches`
//! with the opening-specific noise vector.
//! When `n_load_buses == 0`, Category 4 is empty and has no effect.
//!
//! ## Row index types
//!
//! All indices are stored as `usize` to match the `set_row_bounds` interface
//! signature directly; no casting is required at the call site.
//!
//! ## Worked example (SS4.2a): N = 3, L = 2, no anticipated thermals
//!
//! ```text
//! Row patches (Categories 3 and 5):
//! Patch  Category     Row index   Value
//!     0  noise-fix    3 + 0 = 3   noise[0]  (H0; base_rows[s] = N = 3)
//!     1  noise-fix    3 + 1 = 4   noise[1]  (H1)
//!     2  noise-fix    3 + 2 = 5   noise[2]  (H2)
//!     3  z-inflow     0 + 0 = 0   z_rhs[0]  (H0; z_inflow_row_start = 0)
//!     4  z-inflow     0 + 1 = 1   z_rhs[1]  (H1)
//!     5  z-inflow     0 + 2 = 2   z_rhs[2]  (H2)
//!
//! Column patches (Categories 1 and 2; via fill_col_state_patches):
//! Entry  Category     Column index  Value
//!     0  storage      12 + 0 = 12   state[0]  (H0; storage_in.start = N*(2+L) = 12)
//!     1  storage      12 + 1 = 13   state[1]  (H1)
//!     2  storage      12 + 2 = 14   state[2]  (H2)
//!     3  lag          3 + 0 = 3     state[3]  (H0 lag 0; inflow_lags.start = N = 3)
//!     4  lag          3 + 1 = 4     state[4]  (H1 lag 0)
//!     5  lag          3 + 2 = 5     state[5]  (H2 lag 0)
//!     6  lag          3 + 3 = 6     state[6]  (H0 lag 1)
//!     7  lag          3 + 4 = 7     state[7]  (H1 lag 1)
//!     8  lag          3 + 5 = 8     state[8]  (H2 lag 1)
//! ```
//!
//! Total row patches: 6 = N + N.  Total column patches: 9 = N*(1+L).

use cobre_core::{ConstraintSense, FillingConfig};

mod columns;
mod entries;
mod fpha_cursor;
mod layout;
mod patch;
mod rows;
mod scaling;
mod template;

#[cfg(test)]
mod test_support;

// --- Public re-exports (stable API) ---
pub use patch::PatchBuffer;
pub use template::{StageGeometry, StageTemplates, build_stage_templates};

// --- Crate-internal re-exports ---
pub(crate) use scaling::{apply_col_scale, apply_row_scale, compute_col_scale, compute_row_scale};

// ---------------------------------------------------------------------------
// Commissioning window
// ---------------------------------------------------------------------------

/// Whether an entity is operationally commissioned at `stage_id`.
///
/// An entity is active when the stage is at or after its `entry` (if any) and
/// strictly before its `exit` (if any): `entry <= stage_id < exit`. The exit
/// bound is half-open — a stage equal to `exit` is decommissioned. `None`
/// entry/exit means "no lower/upper bound", so a window-free entity is active
/// at every stage.
///
/// This is the single owner of the commissioning predicate shared by every
/// equipment family. Under the dense layout an inactive entity keeps its LP
/// column at every stage but contributes nothing: callers force its
/// operational bounds to `[0, 0]` (the zero-influence convention) rather than
/// omitting the column. Returning `true`/`false` here, not an active-subset
/// index, is what removes the per-stage active-set bookkeeping and the
/// local↔system column remap; the dense column position is the entity's system
/// index at every stage.
#[inline]
#[must_use]
pub(crate) fn commissioning_active(entry: Option<i32>, exit: Option<i32>, stage_id: i32) -> bool {
    entry.is_none_or(|e| e <= stage_id) && exit.is_none_or(|e| stage_id < e)
}

// ---------------------------------------------------------------------------
// Filling lifecycle phase
// ---------------------------------------------------------------------------

/// Lifecycle phase of a commissioned reservoir at one stage.
///
/// A closed three-variant set the LP builder matches on by enum dispatch (the
/// house rule for a fixed variant set; no trait-object indirection). Derived by
/// [`filling_phase`], which is the single owner of the derivation — see its
/// doc-comment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// Before `start_stage_id`: the dam does not exist; the river flows past its
    /// site. Reached only when `start_stage_id > 0`.
    PreFilling,
    /// `start_stage_id <= stage_id < entry`: the reservoir impounds water toward
    /// the dead volume but is not yet a generating plant.
    Filling,
    /// `stage_id >= entry`, and the all-stages phase of a hydro with no
    /// `FillingConfig`: a normal operating plant.
    Operating,
}

/// The lifecycle [`Phase`] of a hydro at `stage_id`.
///
/// Keyed on `stage_id` — the study **stage id** (`stage.id`), never the stage
/// index. The two coincide for a simple single-resolution horizon but diverge
/// under multi-resolution / decomposition stages; keying on the index there
/// would assign the wrong phase. This mirrors [`commissioning_active`]'s
/// `stage_id: i32` convention.
///
/// `filling.is_none()` ⇒ [`Phase::Operating`] at every stage — the
/// parity-neutrality contract: a hydro with no `FillingConfig` is bit-identical
/// to a normal operating hydro regardless of `entry`. (Entry without filling is
/// rejected upstream by filling validation, so a `None` filling is a normal
/// hydro whatever `entry` holds.)
///
/// With a `FillingConfig { start_stage_id, .. }` and `entry = Some(e)`:
/// `start_stage_id > 0 && stage_id < start_stage_id` ⇒ `PreFilling`;
/// `start_stage_id <= stage_id < e` ⇒ `Filling`; `stage_id >= e` ⇒ `Operating`.
/// `start_stage_id == 0` has **no** `PreFilling` — Filling begins at stage 0
/// (how a study that starts mid-filling is expressed).
///
/// Single owner of the phase derivation. Every per-phase gating site (column
/// bounds, row emission, FPHA exclusion) derives the phase inline by calling
/// this; no caller may cache a per-stage `Phase` mask (the dense-layout
/// no-per-stage-activity-mask rule — like [`commissioning_active`], the phase is
/// recomputed per stage at each site, not stored). A total function: every input
/// maps to a `Phase`, no panic.
///
/// The parameter shape is `Option<&FillingConfig>` + `entry: Option<i32>`
/// (rather than the `(start, entry, stage_id)` scalar triple the design prose
/// sketches) because call sites hold the `Hydro` and pass `hydro.filling.as_ref()`
/// and `hydro.entry_stage_id` directly — taking the config avoids re-deriving
/// `start_stage_id` at every site, matching `commissioning_active`'s `Option`
/// convention.
#[inline]
#[must_use]
pub(crate) fn filling_phase(
    filling: Option<&FillingConfig>,
    entry: Option<i32>,
    stage_id: i32,
) -> Phase {
    let Some(config) = filling else {
        return Phase::Operating;
    };
    if config.start_stage_id > 0 && stage_id < config.start_stage_id {
        return Phase::PreFilling;
    }
    match entry {
        Some(e) if stage_id < e => Phase::Filling,
        _ => Phase::Operating,
    }
}

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

/// Per-hour conversion factor from m³/s to hm³.
///
/// `M3S_TO_HM3 = seconds_per_hour / m³_per_hm³ = 3600 / 1_000_000 = 0.0036`
///
/// Callers multiply by `Block::duration_hours` to get the full block
/// conversion: `volume_hm3 = flow_m3s * M3S_TO_HM3 * duration_hours`.
/// For a 30-day month (720 h): `0.0036 * 720 = 2.592`.
pub(crate) const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;

/// Divisor applied to all objective-function cost coefficients.
///
/// Dividing monetary costs by this factor reduces objective magnitudes
/// (e.g., from ~8.8e11 to ~8.8e8), improving LP solver numerical
/// conditioning without changing the optimization argmin. All cost-domain
/// outputs (objective values, duals, cost breakdowns) are multiplied by
/// this factor at the reporting boundary to recover original units.
pub(crate) const COST_SCALE_FACTOR: f64 = 1_000_000.0;

/// Safety margin applied symmetrically to the magnitude bound on the evaporation
/// outflow variable.  The bound is `[-q_max, +q_max]` where
/// `q_max = |intercept_m3s + volume_slope_m3s_per_hm3 * v_max| * margin`.  A 2x
/// margin accounts for linearization approximation error: the actual area-volume
/// curve may exceed the linear estimate near `v_max` in either direction, so the
/// same factor covers both the positive case (true evaporation underestimated at
/// high storage) and the negative case (net rainfall input underestimated at high
/// storage).  A negative evaporation-outflow value represents net rainfall input
/// (inflow) rather than evaporative outflow.
pub(crate) const EVAPORATION_FLOW_SAFETY_MARGIN: f64 = 2.0;

/// Number of LP columns reserved per evaporating hydro (dimensionless stride).
///
/// Each evaporating hydro contributes exactly three stage-level (NOT per-block)
/// columns in a fixed order: evaporation outflow, `f_evap_plus`, `f_evap_minus`.
/// The base column for local index `i` is `col_evap_start + i * EVAP_COLS_PER_HYDRO`;
/// the three within-hydro columns are reached by adding the offset consts below.
/// The stride MUST be the per-hydro column count and the local index MUST be the
/// outer factor: writing `blk * n_evap_hydros + i` (a transposed/block-major
/// stride) compiles and silently aliases one hydro's evaporation columns onto
/// another's. Owned jointly by [`StageLayout`]'s evaporation accessors and the
/// indexer's `EvaporationIndices` constructor; both reference this const so the
/// stride lives in one place.
///
/// [`StageLayout`]: layout::StageLayout
pub(crate) const EVAP_COLS_PER_HYDRO: usize = 3;

/// Offset of the evaporation-outflow column within a hydro's evaporation block.
///
/// Relative to `col_evap_start + i * EVAP_COLS_PER_HYDRO`; this is the signed
/// evaporation-outflow column (a negative value reads as net rainfall input).
pub(crate) const EVAP_FLOW_OFFSET: usize = 0;

/// Offset of the `f_evap_plus` (under-evaporation) slack column within a hydro's
/// evaporation block.
///
/// Relative to `col_evap_start + i * EVAP_COLS_PER_HYDRO`. Swapping this with
/// [`EVAP_F_MINUS_OFFSET`] compiles and silently misplaces the directional
/// evaporation-violation slacks onto each other's columns.
pub(crate) const EVAP_F_PLUS_OFFSET: usize = 1;

/// Offset of the `f_evap_minus` (over-evaporation) slack column within a hydro's
/// evaporation block.
///
/// Relative to `col_evap_start + i * EVAP_COLS_PER_HYDRO`. Swapping this with
/// [`EVAP_F_PLUS_OFFSET`] compiles and silently misplaces the directional
/// evaporation-violation slacks onto each other's columns.
pub(crate) const EVAP_F_MINUS_OFFSET: usize = 2;

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Per-row metadata for one active generic constraint row at a single stage.
///
/// Stores the information needed by the LP builder to fill CSC matrix entries,
/// row bounds, and objective coefficients for generic constraint rows and their
/// associated slack columns.  Also used by the simulation extraction pipeline
/// to map LP row/column indices back to constraint identity and block.
///
/// One entry is created per active `(constraint, block)` pair, with one
/// exception: a `block_id = None` bound whose expression is **block-independent**
/// (every term resolves to a per-stage column) collapses to a *single*
/// stage-level entry (`is_stage_level = true`) instead of one per block, since
/// the replicated rows would be identical. A `block_id = None` bound on a
/// block-level expression still generates one entry per block; a `block_id =
/// Some(k)` bound generates exactly one entry.
#[derive(Debug, Clone)]
pub struct GenericConstraintRowEntry {
    /// Index into `System::generic_constraints()` for the parent constraint.
    pub constraint_idx: usize,
    /// Entity ID of the parent constraint (copied from `GenericConstraint::id`).
    ///
    /// Stored here so that the simulation extraction pipeline can report the
    /// constraint identity without needing a reference to the full constraint list.
    pub entity_id: i32,
    /// Block index within the stage (0-indexed).
    ///
    /// For a collapsed stage-level row (`is_stage_level = true`) this is the
    /// sentinel `0`: block-independent terms resolve to the same column for any
    /// block, so the marker only needs to be a valid block index.
    pub block_idx: usize,
    /// Whether this row represents the whole stage (a collapsed `block_id = None`
    /// row over a block-independent expression). When `true`, the slack column is
    /// priced by the stage's total hours rather than `block_idx`'s block hours.
    pub is_stage_level: bool,
    /// The right-hand-side bound value for this row.
    pub bound: f64,
    /// Comparison sense of the constraint (`>=`, `<=`, or `==`).
    pub sense: ConstraintSense,
    /// Whether slack is enabled for this constraint.
    pub slack_enabled: bool,
    /// Penalty cost per unit of slack violation (`None` when slack is disabled).
    pub slack_penalty: f64,
    /// Column index of the positive-violation slack (`slack_plus`) when
    /// `slack.enabled = true`.  `None` when slack is disabled.
    pub slack_plus_col: Option<usize>,
    /// Column index of the negative-violation slack (`slack_minus`) when
    /// `slack.enabled = true` and `sense == Equal`.  `None` otherwise.
    pub slack_minus_col: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::{FillingConfig, Phase, filling_phase};

    /// Builds a `FillingConfig` with the given `start_stage_id`; the impound cap
    /// is irrelevant to phase derivation, so any non-negative value is fine.
    fn config(start_stage_id: i32) -> FillingConfig {
        FillingConfig {
            start_stage_id,
            filling_min_rate_m3s: 0.0,
        }
    }

    /// `filling.is_none()` ⇒ `Operating` at every stage, for any `entry` — the
    /// parity-neutrality contract. The forbidden alternative — letting a stray
    /// `entry` push a filling-free hydro into `Filling`/`PreFilling` — would
    /// silently change a normal hydro's physics.
    #[test]
    fn none_filling_is_operating_for_any_entry_and_stage() {
        for entry in [None, Some(0), Some(4)] {
            for stage_id in [-1, 0, 1, 4, 100] {
                assert_eq!(
                    filling_phase(None, entry, stage_id),
                    Phase::Operating,
                    "none filling at entry={entry:?}, stage_id={stage_id}"
                );
            }
        }
    }

    /// With `start_stage_id = 2` and `entry = 4`, the three phases at their exact
    /// transition ids: `stage_id == entry - 1` is the last Filling stage and
    /// `stage_id == entry` is the first Operating stage (the half-open `< entry`
    /// boundary — a non-strict `<= entry` would keep the reservoir Filling one
    /// stage too long).
    #[test]
    fn three_phases_at_exact_boundaries() {
        let f = config(2);
        let entry = Some(4);
        // PreFilling: start_stage_id > 0 and stage_id < start_stage_id.
        assert_eq!(filling_phase(Some(&f), entry, 0), Phase::PreFilling);
        assert_eq!(filling_phase(Some(&f), entry, 1), Phase::PreFilling);
        // Filling: start_stage_id <= stage_id < entry. stage_id == start.
        assert_eq!(filling_phase(Some(&f), entry, 2), Phase::Filling);
        // stage_id == entry - 1 (last Filling stage).
        assert_eq!(filling_phase(Some(&f), entry, 3), Phase::Filling);
        // Operating: stage_id == entry (first Operating stage).
        assert_eq!(filling_phase(Some(&f), entry, 4), Phase::Operating);
        assert_eq!(filling_phase(Some(&f), entry, 5), Phase::Operating);
    }

    /// `start_stage_id == 0` ⇒ no `PreFilling`: Filling runs from stage 0 (how a
    /// study that starts mid-filling is expressed). The forbidden alternative —
    /// treating `stage_id < start_stage_id` without the `start_stage_id > 0`
    /// guard — is moot here (no stage is below 0), but stage 0 must be Filling,
    /// not `PreFilling`.
    #[test]
    fn start_zero_is_filling_at_stage_zero() {
        let f = config(0);
        let entry = Some(4);
        assert_eq!(filling_phase(Some(&f), entry, 0), Phase::Filling);
        assert_eq!(filling_phase(Some(&f), entry, 3), Phase::Filling);
        assert_eq!(filling_phase(Some(&f), entry, 4), Phase::Operating);
    }

    /// A `FillingConfig` with `entry = None` is never Filling/Operating-by-entry:
    /// before `start_stage_id` it is `PreFilling`, at/after it falls through to
    /// `Operating` (the `_` match arm), since there is no entry boundary to cross.
    #[test]
    fn filling_with_none_entry_falls_through_to_operating() {
        let f = config(2);
        assert_eq!(filling_phase(Some(&f), None, 1), Phase::PreFilling);
        assert_eq!(filling_phase(Some(&f), None, 2), Phase::Operating);
        assert_eq!(filling_phase(Some(&f), None, 5), Phase::Operating);
    }
}
