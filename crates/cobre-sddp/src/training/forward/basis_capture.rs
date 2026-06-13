//! Captured-basis metadata population after a forward stage solve.
//!
//! Owns `write_capture_metadata`: records the active-cut slot identities and the
//! state at capture into a `CapturedBasis` so the next iteration's warm-start can
//! match stored cut rows to current LP rows by slot identity.

use crate::cut::pool::CutPool;
use crate::workspace::CapturedBasis;

/// Populate `CapturedBasis` metadata after a stage solve.
///
/// `cut_row_count` is the number of cut rows actually in the LP (derived from
/// `basis_row_capacity - base_row_count`). Under the active-only bake model
/// the baked template carries one row per active cut in `active_cuts()`
/// iteration order; LP row `k` corresponds to the k-th active slot. The
/// metadata captures that identity by pushing each active `slot as u32` in
/// iteration order.
///
/// `row_status` is defensively resized to `base_row_count + cut_row_count` so
/// the metadata invariant holds even when the underlying solver's `get_basis`
/// is a no-op (e.g. test mocks). For real solvers this is a no-op since they
/// write the correct length.
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn write_capture_metadata(
    captured: &mut CapturedBasis,
    pool: &CutPool,
    base_row_count: usize,
    cut_row_count: usize,
    current_state: &[f64],
) {
    captured.cut_row_slots.clear();
    for (slot, _intercept, _coeffs) in pool.active_cuts().take(cut_row_count) {
        captured.cut_row_slots.push(slot as u32);
    }
    captured.state_at_capture.clear();
    captured.state_at_capture.extend_from_slice(current_state);
    captured.base_row_count = base_row_count;
    let expected_len = base_row_count + cut_row_count;
    if captured.basis.row_status.len() != expected_len {
        captured.basis.row_status.resize(
            expected_len,
            crate::basis_reconstruct::HIGHS_BASIS_STATUS_BASIC,
        );
    }
}
