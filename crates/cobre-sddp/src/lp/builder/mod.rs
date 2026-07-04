//! Stage LP patch buffer and stage template builder for SDDP subproblems.
//!
//! - [`PatchBuffer`]: parallel arrays consumed by `set_row_bounds` /
//!   `set_col_bounds`, filled with scenario-dependent values before each LP solve.
//!   Allocated once at training start and reused across every iteration/stage — the
//!   training loop fills it millions of times.
//! - [`build_stage_templates`]: one `StageTemplate` per stage encoding the full
//!   structural LP (CSC matrix, bounds, objective), built once and shared read-only.
//!
//! The column/row geometry — regions, ordering, offset arithmetic — is owned per
//! stage by `StageLayout` (state-vector region on [`crate::indexer::StateLayout`],
//! non-state shape on [`crate::indexer::StudyDimensions`]); this module documents
//! only the per-solve patch sequence layered on top.
//!
//! ## State pinning
//!
//! State pinning lives on **incoming-state columns**, not rows: the LP has no
//! state-fixing row range. Both forward-pass pinning (`set_col_bounds`) and
//! backward-pass cut-subgradient extraction resolve the same column via
//! [`crate::indexer::StateLayout::state_to_lp_incoming_column`].
//!
//! ## Patch sequence
//!
//! Each forward-pass solve writes the row buffer (noise at
//! `base_rows[stage]`, load balance when `n_load_buses > 0`,
//! z-inflow) via `fill_forward_patches` / `fill_load_patches` /
//! `fill_z_inflow_patches`, and the column buffer (incoming storage,
//! AR lags, anticipated state) via `fill_col_state_patches`.
//! The backward pass writes only the column buffer; noise comes from the fixed
//! opening tree through `fill_forward_patches` with the opening-specific vector.

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

/// Whether an entity is operationally commissioned at `stage_id`:
/// `entry <= stage_id < exit`, with the exit bound **half-open** (a stage equal to
/// `exit` is decommissioned) and `None` entry/exit meaning no lower/upper bound.
///
/// Single owner of the commissioning predicate for every equipment family. Returns
/// `true`/`false`, not an active-subset index: under the dense layout an inactive
/// entity keeps its LP column (callers force its bounds to `[0, 0]`), so the column
/// position is the entity's system index at every stage and no per-stage active-set
/// remap is needed.
#[inline]
#[must_use]
pub(crate) fn commissioning_active(entry: Option<i32>, exit: Option<i32>, stage_id: i32) -> bool {
    entry.is_none_or(|e| e <= stage_id) && exit.is_none_or(|e| stage_id < e)
}

// ---------------------------------------------------------------------------
// Filling lifecycle phase
// ---------------------------------------------------------------------------

/// Lifecycle phase of a commissioned reservoir at one stage. Derived solely by
/// [`filling_phase`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// Before `start_stage_id` (only when `start_stage_id > 0`), and every
    /// commissioning-dormant stage of a non-filling hydro: the dam is not built;
    /// the river flows past its site.
    PreFilling,
    /// `start_stage_id <= stage_id < entry`: impounding water toward the dead
    /// volume, not yet a generating plant.
    Filling,
    /// `stage_id >= entry`, and every commissioned stage of a hydro with no
    /// `FillingConfig`: a normal operating plant.
    Operating,
}

/// The lifecycle [`Phase`] of a hydro at `stage_id`.
///
/// Keyed on the study **stage id** (`stage.id`), never the stage index: the two
/// diverge under multi-resolution / decomposition stages, where keying on the index
/// assigns the wrong phase (mirrors [`commissioning_active`]).
///
/// A non-filling hydro (`filling.is_none()`) IS [`Phase::PreFilling`] at every
/// commissioning-dormant stage (`!commissioning_active`) and [`Phase::Operating`]
/// otherwise: the dam is not built, so the river flows past its un-built site via
/// the same short-circuit reformulation a filling hydro uses before `start_stage_id`.
/// The forbidden alternative — zeroing its flow columns while leaving its inflow on
/// its own balance row — traps the water and makes the LP infeasible whenever the
/// site has inflow. `entry/exit = None` ⇒ always `commissioning_active` ⇒
/// `Operating` at every stage, bit-identical to a normal hydro (parity-neutral).
///
/// Single owner of the phase derivation. Every per-phase gating site (column bounds,
/// row emission, FPHA exclusion) recomputes the phase by calling this; no caller may
/// cache a per-stage `Phase` mask (the dense-layout no-per-stage-activity-mask rule).
/// Total function, no panic.
#[inline]
#[must_use]
pub(crate) fn filling_phase(
    filling: Option<&FillingConfig>,
    entry: Option<i32>,
    exit: Option<i32>,
    stage_id: i32,
) -> Phase {
    let Some(config) = filling else {
        return if commissioning_active(entry, exit, stage_id) {
            Phase::Operating
        } else {
            Phase::PreFilling
        };
    };
    if config.start_stage_id > 0 && stage_id < config.start_stage_id {
        return Phase::PreFilling;
    }
    match entry {
        Some(e) if stage_id < e => Phase::Filling,
        _ => Phase::Operating,
    }
}

/// Whether hydro is operationally active (generating) at `stage_id`:
/// `true` iff [`filling_phase`] is [`Phase::Operating`].
///
/// The single source of truth for the per-`(hydro, stage)` active decision,
/// callable outside the LP builder (the policy writer reads it to populate the
/// `was_active` cut-manifest flag). A dormant/`PreFilling`/`Filling` hydro returns
/// `false`. Total function, no panic.
#[inline]
#[must_use]
pub(crate) fn hydro_operating_active(
    filling: Option<&FillingConfig>,
    entry: Option<i32>,
    exit: Option<i32>,
    stage_id: i32,
) -> bool {
    matches!(
        filling_phase(filling, entry, exit, stage_id),
        Phase::Operating
    )
}

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

/// Per-hour conversion factor from m³/s to hm³:
/// `seconds_per_hour / m³_per_hm³ = 3600 / 1_000_000`. Callers multiply by
/// `Block::duration_hours`: `volume_hm3 = flow_m3s * M3S_TO_HM3 * duration_hours`.
pub(crate) const M3S_TO_HM3: f64 = 3_600.0 / 1_000_000.0;

/// Divisor applied to all objective-function cost coefficients to improve LP
/// conditioning without changing the argmin. Every cost-domain output (objective,
/// duals, cost breakdowns) is multiplied back by this factor at the reporting
/// boundary to recover original units.
pub(crate) const COST_SCALE_FACTOR: f64 = 1_000_000.0;

/// Margin on the symmetric magnitude bound `[-q_max, +q_max]` of the evaporation
/// outflow variable, absorbing linearization error where the area-volume curve
/// exceeds the linear estimate near `v_max`. Symmetric because that error runs both
/// directions: a negative evaporation-outflow value is net rainfall input (inflow),
/// a positive one is evaporative outflow.
pub(crate) const EVAPORATION_FLOW_SAFETY_MARGIN: f64 = 2.0;

/// Number of LP columns per `(evaporating hydro, block)` triple: evaporation
/// outflow, `f_evap_plus`, `f_evap_minus`. Base column for evap-local index `i`,
/// block `blk` is `col_evap_start + (i * n_blks + blk) * EVAP_COLS_PER_HYDRO`
/// (hydro-outer, block-middle, offset-inner). The transposed
/// `blk * n_evap_hydros + i` stride compiles and silently aliases one hydro's
/// block onto another's. Single owner of the stride — [`StageLayout`]'s
/// evaporation accessors and the indexer's `EvaporationIndices` constructor both
/// reference this const.
///
/// [`StageLayout`]: layout::StageLayout
pub(crate) const EVAP_COLS_PER_HYDRO: usize = 3;

/// Offset of the signed evaporation-outflow column within a hydro's evaporation
/// block (a negative value reads as net rainfall input).
pub(crate) const EVAP_FLOW_OFFSET: usize = 0;

/// Offset of the `f_evap_plus` (under-evaporation) slack column. Swapping with
/// [`EVAP_F_MINUS_OFFSET`] compiles and silently misplaces the directional
/// evaporation-violation slacks onto each other's columns.
pub(crate) const EVAP_F_PLUS_OFFSET: usize = 1;

/// Offset of the `f_evap_minus` (over-evaporation) slack column. Swapping with
/// [`EVAP_F_PLUS_OFFSET`] compiles and silently misplaces the directional
/// evaporation-violation slacks onto each other's columns.
pub(crate) const EVAP_F_MINUS_OFFSET: usize = 2;

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Per-row metadata for one active generic constraint row at a single stage, used
/// by the LP builder (CSC entries, bounds, objective) and the simulation extraction
/// pipeline (LP index → constraint identity + block).
///
/// One entry per active `(constraint, block)` pair, except: a `block_id = None`
/// bound whose expression is **block-independent** collapses to a *single*
/// stage-level entry (`is_stage_level = true`), since the replicated rows would be
/// identical. A `block_id = None` bound on a block-level expression still generates
/// one entry per block; a `block_id = Some(k)` bound generates exactly one.
#[derive(Debug, Clone)]
pub struct GenericConstraintRowEntry {
    /// Index into `System::generic_constraints()` for the parent constraint.
    pub constraint_idx: usize,
    /// Entity ID of the parent constraint (copied from `GenericConstraint::id`).
    pub entity_id: i32,
    /// Block index within the stage (0-indexed); the sentinel `0` for a collapsed
    /// stage-level row (`is_stage_level = true`), which resolves the same column for
    /// any block.
    pub block_idx: usize,
    /// Whether this row is a collapsed stage-level row; when `true` the slack column
    /// is priced by the stage's total hours, not `block_idx`'s block hours.
    pub is_stage_level: bool,
    /// The right-hand-side bound value for this row.
    pub bound: f64,
    /// Comparison sense of the constraint (`>=`, `<=`, or `==`).
    pub sense: ConstraintSense,
    /// Whether slack is enabled for this constraint.
    pub slack_enabled: bool,
    /// Penalty cost per unit of slack violation (`None` when slack is disabled).
    pub slack_penalty: f64,
    /// Positive-violation slack (`slack_plus`) column; `None` when slack is disabled.
    pub slack_plus_col: Option<usize>,
    /// Negative-violation slack (`slack_minus`) column, present only when slack is
    /// enabled and `sense == Equal`; `None` otherwise.
    pub slack_minus_col: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::{FillingConfig, Phase, filling_phase, hydro_operating_active};

    /// Builds a `FillingConfig` with the given `start_stage_id`; the impound cap
    /// is irrelevant to phase derivation, so any non-negative value is fine.
    fn config(start_stage_id: i32) -> FillingConfig {
        FillingConfig {
            start_stage_id,
            filling_min_rate_m3s: 0.0,
        }
    }

    /// `filling.is_none()` with `entry/exit = None` ⇒ `Operating` at every stage —
    /// the parity-neutrality contract: an un-commissioned-window hydro is
    /// bit-identical to a normal operating hydro. The forbidden alternative — letting
    /// the new dormant branch push a window-free hydro into `PreFilling` — would
    /// change every existing D-case's physics.
    #[test]
    fn none_filling_none_window_is_operating_at_every_stage() {
        for stage_id in [-1, 0, 1, 4, 100] {
            assert_eq!(
                filling_phase(None, None, None, stage_id),
                Phase::Operating,
                "none filling, no window, stage_id={stage_id}"
            );
        }
    }

    /// `filling.is_none()` with a commissioning window: `PreFilling` before `entry`,
    /// `Operating` from `entry` (no intervening `Filling` — a non-filling hydro has
    /// no impounding stage), then `PreFilling` again at/after `exit`. The forbidden
    /// alternative — `Operating` before `entry` — would leave the un-built dam's
    /// inflow trapped on its own balance row.
    #[test]
    fn none_filling_with_window_is_prefilling_outside_entry_exit() {
        let entry = Some(2);
        let exit = Some(5);
        assert_eq!(filling_phase(None, entry, exit, 0), Phase::PreFilling);
        assert_eq!(filling_phase(None, entry, exit, 1), Phase::PreFilling);
        // First commissioned stage: straight to Operating, no Filling.
        assert_eq!(filling_phase(None, entry, exit, 2), Phase::Operating);
        assert_eq!(filling_phase(None, entry, exit, 4), Phase::Operating);
        // exit is half-open: stage_id == exit is decommissioned (dormant again).
        assert_eq!(filling_phase(None, entry, exit, 5), Phase::PreFilling);
        assert_eq!(filling_phase(None, entry, exit, 9), Phase::PreFilling);
    }

    /// `filling.is_none()` with `entry` beyond the horizon ⇒ `PreFilling` at every
    /// stage (always-dormant), the "plant never commissions in this study" case.
    #[test]
    fn none_filling_entry_beyond_horizon_is_prefilling_everywhere() {
        let entry = Some(1000);
        for stage_id in [0, 1, 4, 100] {
            assert_eq!(
                filling_phase(None, entry, None, stage_id),
                Phase::PreFilling,
                "always-dormant non-filling at stage_id={stage_id}"
            );
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
        assert_eq!(filling_phase(Some(&f), entry, None, 0), Phase::PreFilling);
        assert_eq!(filling_phase(Some(&f), entry, None, 1), Phase::PreFilling);
        // Filling: start_stage_id <= stage_id < entry. stage_id == start.
        assert_eq!(filling_phase(Some(&f), entry, None, 2), Phase::Filling);
        // stage_id == entry - 1 (last Filling stage).
        assert_eq!(filling_phase(Some(&f), entry, None, 3), Phase::Filling);
        // Operating: stage_id == entry (first Operating stage).
        assert_eq!(filling_phase(Some(&f), entry, None, 4), Phase::Operating);
        assert_eq!(filling_phase(Some(&f), entry, None, 5), Phase::Operating);
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
        assert_eq!(filling_phase(Some(&f), entry, None, 0), Phase::Filling);
        assert_eq!(filling_phase(Some(&f), entry, None, 3), Phase::Filling);
        assert_eq!(filling_phase(Some(&f), entry, None, 4), Phase::Operating);
    }

    /// A `FillingConfig` with `entry = None` is never Filling/Operating-by-entry:
    /// before `start_stage_id` it is `PreFilling`, at/after it falls through to
    /// `Operating` (the `_` match arm), since there is no entry boundary to cross.
    #[test]
    fn filling_with_none_entry_falls_through_to_operating() {
        let f = config(2);
        assert_eq!(filling_phase(Some(&f), None, None, 1), Phase::PreFilling);
        assert_eq!(filling_phase(Some(&f), None, None, 2), Phase::Operating);
        assert_eq!(filling_phase(Some(&f), None, None, 5), Phase::Operating);
    }

    /// `hydro_operating_active` truth table — the per-(hydro, stage) active predicate
    /// the policy writer reads for `was_active`: `true` iff `Operating`, `false` for
    /// `PreFilling`/`Filling`. Pins each branch (non-filling no-window, non-filling
    /// dormant before/at/after window, filling lifecycle).
    #[test]
    fn hydro_operating_active_truth_table() {
        // Non-filling, no window: active everywhere.
        assert!(hydro_operating_active(None, None, None, 0));
        assert!(hydro_operating_active(None, None, None, 100));
        // Non-filling, window [2, 5): dormant before entry, active inside, dormant
        // from exit.
        assert!(!hydro_operating_active(None, Some(2), Some(5), 1));
        assert!(hydro_operating_active(None, Some(2), Some(5), 2));
        assert!(hydro_operating_active(None, Some(2), Some(5), 4));
        assert!(!hydro_operating_active(None, Some(2), Some(5), 5));
        // Filling lifecycle (start 2, entry 4): inactive in PreFilling and Filling,
        // active from entry.
        let f = config(2);
        assert!(!hydro_operating_active(Some(&f), Some(4), None, 0)); // PreFilling
        assert!(!hydro_operating_active(Some(&f), Some(4), None, 3)); // Filling
        assert!(hydro_operating_active(Some(&f), Some(4), None, 4)); // Operating
    }
}
