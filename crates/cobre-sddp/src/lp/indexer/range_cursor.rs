//! The [`RangeCursor`] running column/row offset allocator shared by
//! [`StageLayout::new`](crate::lp_builder::StageLayout)'s per-stage equipment
//! column/row chains and [`StateLayout::new`](super::StateLayout)'s
//! stage-invariant state-vector chain.

use std::ops::Range;

/// A running column/row offset allocator: [`Self::alloc`] returns `pos..pos +
/// len` and advances the cursor by `len`, so a family's start is never
/// re-threaded by hand and adjacency between consecutive families — the next
/// family's start equals the previous family's end — is structural, not a
/// hand-copied `.end`.
///
/// `alloc(0)` returns `pos..pos`, the live cursor position, never `0..0` —
/// `0..0` loses the position an empty-block-cursor field
/// (`generation_col_start`/`evap_col_start`/`post_equipment_col_start`/
/// `post_equipment_row_start`) or an `n_h == 0` accessor fallback needs;
/// `pos..pos` carries it, so those reads and fallbacks collapse to a bare
/// `.start`/`.end` with no branch. A caller that needs the literal `0..0`
/// convention for an optional block (e.g.
/// [`StateLayout::new`](super::StateLayout)) normalises it explicitly at the
/// call site — `RangeCursor` itself never returns `0..0`.
pub(crate) struct RangeCursor {
    pos: usize,
}

impl RangeCursor {
    pub(crate) fn new(start: usize) -> Self {
        Self { pos: start }
    }

    pub(crate) fn alloc(&mut self, len: usize) -> Range<usize> {
        let start = self.pos;
        self.pos += len;
        start..self.pos
    }

    pub(crate) fn pos(&self) -> usize {
        self.pos
    }
}
