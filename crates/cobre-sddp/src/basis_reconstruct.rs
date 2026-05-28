//! Slot-tracked basis reconstruction for cut-set-aware warm-start.
//!
//! This module provides [`reconstruct_basis`] — the slot-identity-based helper
//! that correctly handles cut-set churn (drops, reorders, adds) between
//! iterations.
//!
//! ## Why slot identity matters
//!
//! Length-keyed basis reconciliation (matching stored row statuses to the
//! current LP by row count alone) breaks under cut-set churn: when cut
//! selection replaces one cut with another of equal count, the two bases are
//! the same length but positionally misaligned. `HiGHS` would then receive a
//! basis with mismatched row statuses and either reject the basis (cold
//! start) or warm-start with a corrupted basis.
//!
//! [`reconstruct_basis`] takes [`CapturedBasis::cut_row_slots`] as its key:
//! each stored cut row carries the [`CutPool`](crate::cut::pool::CutPool) slot
//! that generated it. The reconstruction walks the current LP's cut rows in
//! order, looks each slot up in an O(1) scratch map, and either copies the
//! stored status (slot found → preserved) or assigns
//! [`HIGHS_BASIS_STATUS_BASIC`] (slot not found → new cut).
//!
//! ## Why new cuts default to BASIC
//!
//! `HiGHS` requires `col_basic + row_basic == num_row` for any warm-start
//! basis. Assigning `BASIC` to every new cut row preserves the invariant by
//! construction: each new cut adds exactly one row and exactly one `BASIC`
//! count, so the equality balances on both sides. Classifying a new cut as
//! `LOWER` would break the equality and force a compensating demotion
//! elsewhere.
//!
//! ## Forward-path basic-count invariant
//!
//! On the forward path, cut selection may drop cuts whose stored row status
//! was `HIGHS_BASIS_STATUS_BASIC`. Each such BASIC drop leaves one fewer
//! BASIC row. [`reconstruct_basis`] still assigns `BASIC` unconditionally to
//! new cuts, so the reconstructed basis may temporarily carry an `excess`
//! of basic statuses (`excess = col_basic + row_basic - num_row > 0`).
//!
//! [`enforce_basic_count_invariant`] is an unconditional safety net invoked
//! at the single call site after every reconstruction. It scans
//! `out.row_status` from the end and demotes trailing `BASIC` cut-row
//! entries (indices `>= base_row_count`) to `LOWER` until the invariant
//! holds. It is a no-op when `excess == 0`.
//!
//! ## Usage
//!
//! ```rust
//! use cobre_sddp::basis_reconstruct::{
//!     ReconstructionStats, ReconstructionTarget, reconstruct_basis,
//! };
//! use cobre_sddp::workspace::CapturedBasis;
//! use cobre_solver::Basis;
//!
//! let stored = CapturedBasis::new(4, 3, 3, 0, 0); // empty — shim state
//! let target = ReconstructionTarget { base_row_count: 3, num_cols: 4 };
//! let mut out = Basis::new(0, 0);
//! let mut lookup: Vec<Option<u32>> = vec![None; 16];
//! let cuts: Vec<(usize, f64, Vec<f64>)> = vec![];
//! let stats = reconstruct_basis(
//!     &stored,
//!     target,
//!     cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
//!     &mut out,
//!     &mut lookup,
//! );
//! assert_eq!(stats, ReconstructionStats::default());
//! ```

pub use cobre_solver::ffi::{HIGHS_BASIS_STATUS_BASIC, HIGHS_BASIS_STATUS_LOWER};

use cobre_solver::Basis;

use crate::workspace::CapturedBasis;

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

/// Default value for the `basis_activity_window` configuration field.
///
/// Configurable at runtime via `training.cut_selection.basis_activity_window`
/// in the TOML config; validated range 1..=31. This is the fallback value
/// used when the TOML field is absent.
///
/// The field is currently unused by [`reconstruct_basis`] (slot-identity
/// matching makes the activity window unnecessary). It remains as a config
/// surface and a wire-format placeholder pending its deprecation cycle.
pub const DEFAULT_BASIS_ACTIVITY_WINDOW: u32 = 5;

// ---------------------------------------------------------------------------
// Target LP shape
// ---------------------------------------------------------------------------

/// Dimensions of the target LP that [`reconstruct_basis`] populates a basis
/// for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReconstructionTarget {
    /// Number of template (non-cut) rows in the target LP.
    pub base_row_count: usize,
    /// Total column count of the target LP.
    pub num_cols: usize,
}

// ---------------------------------------------------------------------------
// Return type
// ---------------------------------------------------------------------------

/// Counters returned by [`reconstruct_basis`].
///
/// The invariant `preserved + new_tight + new_slack` equals the number of
/// elements the iterator passed to [`reconstruct_basis`] yielded (i.e. the
/// cut-row count of the target LP).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReconstructionStats {
    /// Cut rows whose slot was found in the stored basis and whose status
    /// was copied directly.
    pub preserved: u32,
    /// Always zero with slot-identity classification (kept for telemetry
    /// stability against downstream consumers).
    pub new_tight: u32,
    /// Cut rows whose slot was not present in the stored basis. Each such
    /// row is assigned `HIGHS_BASIS_STATUS_BASIC` to preserve the
    /// `col_basic + row_basic == num_row` invariant by construction.
    pub new_slack: u32,
}

// ---------------------------------------------------------------------------
// reconstruct_basis
// ---------------------------------------------------------------------------

/// Reconstruct a full [`Basis`] for the target LP using slot identity.
///
/// Preserves cut rows already present in the stored basis (copies their
/// stored status verbatim) and assigns [`HIGHS_BASIS_STATUS_BASIC`] to every
/// new cut row.
///
/// ## Parameters
///
/// - `stored` — read-only stored basis from the previous iteration.
/// - `target` — dimensions of the target LP.
/// - `current_cut_rows` — iterator of `(slot, intercept, coefficients)` in
///   target LP row order. The `intercept` and `coefficients` items are
///   accepted for signature parity with the cut-pool iterator but are not
///   consulted: classification depends solely on slot identity.
/// - `out` — destination basis (caller owns; cleared and refilled in place).
/// - `slot_lookup` — scratch `Vec<Option<u32>>` pre-sized by the caller to
///   at least `max_slot + 1`. Grown in place if undersized (hot path should
///   avoid this via `ScratchBuffers::recon_slot_lookup`).
///
/// ## Returns
///
/// [`ReconstructionStats`] with `preserved + new_tight + new_slack` equal to
/// the number of items yielded by the iterator. With slot-identity
/// classification, `new_tight` is always zero.
///
/// ## Allocation contract
///
/// Allocation-free when `slot_lookup.len() >= max_slot + 1`. The growth
/// branch triggers `debug_assert!(false)` to surface caller under-sizing
/// without panicking in release.
pub fn reconstruct_basis<'a, I>(
    stored: &CapturedBasis,
    target: ReconstructionTarget,
    current_cut_rows: I,
    out: &mut Basis,
    slot_lookup: &mut Vec<Option<u32>>,
) -> ReconstructionStats
where
    I: Iterator<Item = (usize, f64, &'a [f64])>,
{
    debug_assert!(
        stored.basis.row_status.len() == stored.base_row_count + stored.cut_row_slots.len(),
        "CapturedBasis invariant violated: row_status.len() {} != base_row_count {} + \
         cut_row_slots.len() {}",
        stored.basis.row_status.len(),
        stored.base_row_count,
        stored.cut_row_slots.len(),
    );

    reconstruct_col_statuses(stored, target, out);
    reconstruct_template_row_statuses(stored, target, out);
    build_slot_lookup(stored.cut_row_slots.as_slice(), slot_lookup);

    let mut stats = ReconstructionStats::default();
    for (target_slot, _intercept, _coefficients) in current_cut_rows {
        let row_status_byte = if let Some(pos) = slot_lookup.get(target_slot).and_then(|o| *o) {
            let stored_row_idx = stored.base_row_count + pos as usize;
            stats.preserved += 1;
            stored.basis.row_status[stored_row_idx]
        } else {
            stats.new_slack += 1;
            HIGHS_BASIS_STATUS_BASIC
        };
        out.row_status.push(row_status_byte);
    }

    stats
}

// ---------------------------------------------------------------------------
// Phase helpers (private — not part of the public API)
// ---------------------------------------------------------------------------

/// Copy column statuses from the stored basis, resizing to match the target
/// column count.
///
/// Copies `stored.basis.col_status` verbatim into `out.col_status`, then
/// extends or truncates to exactly `target.num_cols` entries, padding with
/// `HIGHS_BASIS_STATUS_BASIC` if the target is wider than the stored basis.
fn reconstruct_col_statuses(stored: &CapturedBasis, target: ReconstructionTarget, out: &mut Basis) {
    out.col_status.clear();
    out.col_status.extend_from_slice(&stored.basis.col_status);
    if out.col_status.len() != target.num_cols {
        out.col_status
            .resize(target.num_cols, HIGHS_BASIS_STATUS_BASIC);
    }
}

/// Copy the first `target.base_row_count` template row statuses from the
/// stored basis.
///
/// If the stored basis has fewer template rows than `target.base_row_count`,
/// missing rows are filled with `HIGHS_BASIS_STATUS_BASIC`. Cut rows
/// (indices `>= base_row_count`) are not written here; they are assigned by
/// the slot-identity loop in [`reconstruct_basis`].
fn reconstruct_template_row_statuses(
    stored: &CapturedBasis,
    target: ReconstructionTarget,
    out: &mut Basis,
) {
    out.row_status.clear();
    if stored.basis.row_status.len() >= target.base_row_count {
        out.row_status
            .extend_from_slice(&stored.basis.row_status[..target.base_row_count]);
    } else {
        // Stored basis has fewer template rows than the target — fill missing with BASIC.
        out.row_status.extend_from_slice(&stored.basis.row_status);
        out.row_status
            .resize(target.base_row_count, HIGHS_BASIS_STATUS_BASIC);
    }
}

/// Build the slot → reconcilable-position lookup table.
///
/// Fills `slot_lookup` so that `slot_lookup[slot] = Some(position)` for each
/// slot in `reconcilable_slots` (position is the 0-based index within that
/// slice). Grows `slot_lookup` defensively if it is undersized (should not
/// happen on the hot path when the caller pre-sizes via `ScratchBuffers`).
fn build_slot_lookup(reconcilable_slots: &[u32], slot_lookup: &mut Vec<Option<u32>>) {
    // Grow the scratch if it is too small for the current stored slots. In
    // normal operation the caller pre-sizes this to `initial_pool_capacity`
    // (via `ScratchBuffers`), so growth only occurs on cold paths. When
    // reconcilable_slots is empty there is nothing to look up — skip the
    // size check (an empty lookup is correct in that case).
    if let Some(max_slot_val) = reconcilable_slots.iter().copied().max() {
        let max_slot = max_slot_val as usize;
        if slot_lookup.len() <= max_slot {
            debug_assert!(
                false,
                "slot_lookup undersized ({} <= max_slot {}); caller must pre-size to \
                 initial_pool_capacity",
                slot_lookup.len(),
                max_slot,
            );
            // Defensive growth in release so the function remains safe.
            slot_lookup.resize(max_slot + 1, None);
        }
    }
    slot_lookup.fill(None);
    // `position` here is the index within reconcilable_slots (0-based), so the
    // stored row index is `stored.base_row_count + position`.
    #[allow(clippy::cast_possible_truncation)]
    for (position, &slot) in reconcilable_slots.iter().enumerate() {
        slot_lookup[slot as usize] = Some(position as u32);
    }
}

// ---------------------------------------------------------------------------
// enforce_basic_count_invariant
// ---------------------------------------------------------------------------

/// Restore `col_basic + row_basic == num_row` after [`reconstruct_basis`]
/// on the forward path.
///
/// ## Algebraic invariant
///
/// Let `excess = col_basic + row_basic - num_row` after reconstruction.
///
/// On the forward path, cut selection may drop cuts whose stored row status
/// was `HIGHS_BASIS_STATUS_BASIC`. Each such BASIC drop leaves one fewer
/// BASIC row, but [`reconstruct_basis`] assigns BASIC unconditionally to
/// new cuts. If more cuts were preserved with BASIC status than the LP now
/// has room for, `excess > 0`.
///
/// The investigation proved that `excess = dropped_basic >= 0` always.
/// There is never a deficit. Therefore:
///
/// - If `excess == 0`: no-op, return 0.
/// - If `excess > 0`: scan `out.row_status` from the end, flipping
///   `HIGHS_BASIS_STATUS_BASIC` to `HIGHS_BASIS_STATUS_LOWER` only for
///   cut rows (indices `>= base_row_count`) until exactly `excess` demotions
///   have been applied.
///
/// ## Parameters
///
/// - `out` — the [`Basis`] returned by [`reconstruct_basis`].
/// - `num_row` — the total row count of the target LP
///   (`base_row_count + active_cut_count`).
/// - `base_row_count` — the number of template (non-cut) rows. Only rows
///   at indices `>= base_row_count` are eligible for demotion.
///
/// ## Returns
///
/// The number of demotions applied (0 in the no-op case).
///
/// ## Assertions
///
/// `debug_assert!` checks that `num_row == out.row_status.len()` and
/// `base_row_count <= num_row`.
pub fn enforce_basic_count_invariant(
    out: &mut Basis,
    num_row: usize,
    base_row_count: usize,
) -> u32 {
    debug_assert_eq!(
        num_row,
        out.row_status.len(),
        "enforce_basic_count_invariant: num_row ({num_row}) != out.row_status.len() ({})",
        out.row_status.len(),
    );
    debug_assert!(
        base_row_count <= num_row,
        "enforce_basic_count_invariant: base_row_count ({base_row_count}) > num_row ({num_row})",
    );

    let col_basic = out
        .col_status
        .iter()
        .filter(|&&s| s == HIGHS_BASIS_STATUS_BASIC)
        .count();
    let row_basic = out
        .row_status
        .iter()
        .filter(|&&s| s == HIGHS_BASIS_STATUS_BASIC)
        .count();

    let total_basic = col_basic + row_basic;
    if total_basic <= num_row {
        // Invariant holds (or excess == 0): nothing to do.
        return 0;
    }

    let mut excess = total_basic - num_row;
    let mut demotions: u32 = 0;

    // Scan from the end of row_status, touching only cut rows (>= base_row_count).
    for idx in (base_row_count..out.row_status.len()).rev() {
        if excess == 0 {
            break;
        }
        if out.row_status[idx] == HIGHS_BASIS_STATUS_BASIC {
            out.row_status[idx] = HIGHS_BASIS_STATUS_LOWER;
            excess -= 1;
            demotions += 1;
        }
    }

    demotions
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::doc_markdown)]
mod tests {
    use cobre_solver::Basis;

    use super::{
        HIGHS_BASIS_STATUS_BASIC as B, HIGHS_BASIS_STATUS_LOWER as L, ReconstructionStats,
        ReconstructionTarget, reconstruct_basis,
    };
    use crate::workspace::CapturedBasis;

    /// Build a `CapturedBasis` populated with the requested slot list and
    /// cut-row status sequence. Template rows are all `BASIC`; columns are
    /// all `BASIC` so the column block does not perturb the test focus.
    fn make_stored_basis(
        base_rows: usize,
        num_cols: usize,
        slots: &[u32],
        cut_statuses: &[i32],
        state_at_capture: &[f64],
    ) -> CapturedBasis {
        assert_eq!(slots.len(), cut_statuses.len());
        let total_rows = base_rows + cut_statuses.len();
        let mut cb = CapturedBasis::new(
            num_cols,
            total_rows,
            base_rows,
            slots.len(),
            state_at_capture.len(),
        );

        // Fill basis row_status: template rows = BASIC, then cut statuses.
        cb.basis.row_status.clear();
        cb.basis.row_status.resize(base_rows, B);
        cb.basis.row_status.extend_from_slice(cut_statuses);

        // Fill col_status: all BASIC.
        cb.basis.col_status.clear();
        cb.basis.col_status.resize(num_cols, B);

        // Slot and state metadata.
        cb.cut_row_slots.extend_from_slice(slots);
        cb.state_at_capture.extend_from_slice(state_at_capture);

        cb
    }

    // -----------------------------------------------------------------------
    // reconstruct_basis — unit tests
    // -----------------------------------------------------------------------

    /// Empty stored slot list + 3 new cut rows. Every new row is classified
    /// `BASIC` and `stats.new_slack == 3`.
    #[test]
    fn returns_basic_for_all_new_cuts() {
        let stored = make_stored_basis(1, 2, &[], &[], &[1.0]);
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (5, 0.0, vec![0.0, 0.0]),
            (6, 0.0, vec![0.0, 0.0]),
            (7, 0.0, vec![0.0, 0.0]),
        ];
        let target = ReconstructionTarget {
            base_row_count: 1,
            num_cols: 2,
        };
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 16];

        let stats = reconstruct_basis(
            &stored,
            target,
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            &mut out,
            &mut lookup,
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 0,
                new_tight: 0,
                new_slack: 3,
            },
        );
        // 1 template row + 3 cut rows.
        assert_eq!(out.row_status.len(), 4);
        assert_eq!(&out.row_status[1..], &[B, B, B]);
    }

    /// Two preserved slots — their stored row statuses must be copied
    /// verbatim even when the stored value is `LOWER`. The reconstruction
    /// performs no classifier work, so it must not promote `LOWER` to
    /// `BASIC` itself.
    #[test]
    fn copies_stored_status_for_preserved_slots() {
        let stored = make_stored_basis(1, 2, &[10, 20], &[B, L], &[1.0]);
        let cuts: Vec<(usize, f64, Vec<f64>)> =
            vec![(10, 0.0, vec![0.0, 0.0]), (20, 0.0, vec![0.0, 0.0])];
        let target = ReconstructionTarget {
            base_row_count: 1,
            num_cols: 2,
        };
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 32];

        let stats = reconstruct_basis(
            &stored,
            target,
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            &mut out,
            &mut lookup,
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 2,
                new_tight: 0,
                new_slack: 0,
            },
        );
        assert_eq!(out.row_status.len(), 3);
        assert_eq!(out.row_status[1], B, "slot 10 → stored BASIC");
        assert_eq!(out.row_status[2], L, "slot 20 → stored LOWER");
    }

    /// Mixed case: stored preserves slots `{10, 20, 30, 40}` with statuses
    /// `[L, B, L, B]`; the target LP has 5 cut rows for slots
    /// `{10, 25, 30, 45, 50}`. Slots 10 and 30 are preserved (their stored
    /// statuses copied); slots 25, 45, 50 are new and receive `BASIC`.
    #[test]
    fn mixed_case_preserved_and_new() {
        let stored = make_stored_basis(2, 3, &[10, 20, 30, 40], &[L, B, L, B], &[1.0, 2.0]);
        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![
            (10, 0.0, vec![0.0, 0.0]),
            (25, 0.0, vec![0.0, 0.0]),
            (30, 0.0, vec![0.0, 0.0]),
            (45, 0.0, vec![0.0, 0.0]),
            (50, 0.0, vec![0.0, 0.0]),
        ];
        let target = ReconstructionTarget {
            base_row_count: 2,
            num_cols: 3,
        };
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 64];

        let stats = reconstruct_basis(
            &stored,
            target,
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            &mut out,
            &mut lookup,
        );

        assert_eq!(
            stats,
            ReconstructionStats {
                preserved: 2,
                new_tight: 0,
                new_slack: 3,
            },
            "preserved={{10, 30}}, new_slack={{25, 45, 50}}",
        );
        // 2 template rows + 5 cut rows.
        assert_eq!(out.row_status.len(), 7);
        // Cut row block starts at index 2.
        assert_eq!(out.row_status[2], L, "slot 10 → stored LOWER");
        assert_eq!(out.row_status[3], B, "slot 25 → new → BASIC");
        assert_eq!(out.row_status[4], L, "slot 30 → stored LOWER");
        assert_eq!(out.row_status[5], B, "slot 45 → new → BASIC");
        assert_eq!(out.row_status[6], B, "slot 50 → new → BASIC");
    }

    /// Empty iterator: the cut-row block must be empty and stats must remain
    /// at zero. The template-row block is still populated from the stored
    /// basis.
    #[test]
    fn empty_iterator_preserves_template_rows() {
        let stored = make_stored_basis(3, 2, &[10, 20], &[B, L], &[1.0]);
        let target = ReconstructionTarget {
            base_row_count: 3,
            num_cols: 2,
        };
        let mut out = Basis::new(0, 0);
        let mut lookup: Vec<Option<u32>> = vec![None; 32];

        let cuts: Vec<(usize, f64, Vec<f64>)> = vec![];
        let stats = reconstruct_basis(
            &stored,
            target,
            cuts.iter().map(|(s, i, c)| (*s, *i, c.as_slice())),
            &mut out,
            &mut lookup,
        );

        assert_eq!(stats, ReconstructionStats::default());
        // Only the 3 template rows remain in the output basis.
        assert_eq!(out.row_status.len(), 3);
        assert!(
            out.row_status.iter().all(|&s| s == B),
            "template rows must be copied verbatim (all BASIC in this fixture)",
        );
    }
}
