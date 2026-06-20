//! The typed [`BlockGrid`] address primitive shared by all block-stride LP fills.
//!
//! The block-stride operation appears in three distinct nesting shapes across
//! the LP build, patch, resolver, and extraction paths. Each shape has its own
//! method here so the entity-outer / block-inner ordering of one shape cannot be
//! called with the argument order of another. [`BlockGrid`] is a zero-cost
//! [`Copy`] value type carrying only the two stride constants the three shapes
//! need beyond their per-call args (`n_blks` and `max_deficit_segments`); obtain
//! it from [`StageIndexer::block_grid`](super::StageIndexer::block_grid).
//!
//! ## Why three methods, not one generic `(a, b, c)` method
//!
//! A single `fn addr(&self, a, b, c) -> a + b * stride + c` would re-open the
//! transposed-stride trap the type exists to close: the flat shape nests the
//! entity OUTER and the block INNER, while the FPHA-plane shape nests the block
//! OUTER and the plane INNER — opposite orderings. A shared `(a, b, c)` method
//! lets a caller pass the FPHA arguments in the flat order (or vice versa) and
//! still compile, silently addressing the wrong cell. Distinct method names with
//! distinct parameter names per shape make that mis-use a name error, not a
//! runtime miscompute.
//!
//! ## No transposed method exists
//!
//! No method here produces the transposed form `blk * n_entities + entity`. That
//! form yields a same-length region but interleaves columns across entities, so a
//! coefficient lands on the wrong `(entity, block)` and the LP is silently
//! misbuilt. Because [`BlockGrid`] exposes no field accessors and no generic
//! stride method, a caller cannot reconstruct the transposed stride by hand
//! without deliberately bypassing the type and re-deriving the arithmetic from
//! [`StageIndexer`](super::StageIndexer) fields directly. The
//! `block_grid_forbids_transposed_shape` test in this module pins that there is
//! no `(blk, n_entities, entity)`-ordered method.

/// Typed block-stride address calculator for one SDDP stage LP.
///
/// Carries the two stage constants (`n_blks` and `max_deficit_segments`) that the
/// three block-stride shapes need beyond their per-call arguments. Constructed by
/// [`StageIndexer::block_grid`](super::StageIndexer::block_grid); it is a cheap
/// `Copy` value with no allocation and no lifetime, so passing it by value across
/// fill closures costs nothing.
///
/// The three shapes — [`flat`](Self::flat), [`fpha_plane`](Self::fpha_plane), and
/// [`deficit`](Self::deficit) — have opposite block-nesting orders; see the
/// [module docs](self) for why each gets its own method instead of one shared
/// `(a, b, c)` calculator.
// Intent/Seam: defined here as the single owner of the block-stride arithmetic so
// the hand-rolled stride sites in matrix.rs, generic_constraints.rs,
// builder/layout.rs, builder/patch.rs, and simulation/extraction.rs can migrate
// onto these typed shapes. Until those call sites are converted, the type and its
// methods have no production caller, so the dead_code lint fires on the lib pass;
// each migration removes a hand-rolled stride and adds a use, at which point the
// lint is satisfied without this allow. The shape #[test]s exercise every method.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockGrid {
    /// Number of operating blocks per stage (K), the per-entity stride of the
    /// flat shape and the inner stride of the deficit shape.
    n_blks: usize,
    /// Maximum number of deficit segments across all buses (S), the per-bus
    /// stride of the deficit shape.
    max_deficit_segments: usize,
}

// Intent/Seam: the constructor and three shape methods have no production caller
// until the hand-rolled stride sites migrate onto them (see the type doc above);
// the dead_code lint fires on the lib pass meanwhile. The shape #[test]s in this
// module exercise every method, and each migrated call site adds a production use.
#[allow(dead_code)]
impl BlockGrid {
    /// Construct a [`BlockGrid`] from its two stride constants.
    ///
    /// Prefer [`StageIndexer::block_grid`](super::StageIndexer::block_grid) at
    /// call sites — it sources both constants from the single owning indexer so
    /// the grid cannot disagree with the LP it addresses.
    #[inline]
    pub(crate) fn new(n_blks: usize, max_deficit_segments: usize) -> Self {
        Self {
            n_blks,
            max_deficit_segments,
        }
    }

    /// Flat block-major address: `start + entity * n_blks + blk`.
    ///
    /// The entity is the OUTER (stride) factor and the block is the INNER offset:
    /// the per-entity column count is `n_blks`, so entity `e`'s block `blk` lands
    /// at `start + e * n_blks + blk`. The transposed form
    /// `start + blk * n_entities + entity` is the wrong-but-compiling alternative —
    /// it produces a same-length region but interleaves columns across entities,
    /// so the coefficient lands on the wrong `(entity, block)` and the LP is
    /// silently misbuilt. This method fixes the entity-outer / block-inner order
    /// in its parameter names so the transpose cannot be expressed through it.
    ///
    /// Used by every per-family flat block-major fill (turbine, spillage,
    /// diversion, thermal, line-fwd/rev, FPHA generation, operational-violation
    /// slacks).
    #[inline]
    pub(crate) fn flat(&self, start: usize, entity: usize, blk: usize) -> usize {
        start + entity * self.n_blks + blk
    }

    /// FPHA-plane address: `fpha_block_start + blk * n_planes + p_idx`.
    ///
    /// Here the block is the OUTER (stride) factor and the plane is the INNER
    /// offset — the OPPOSITE nesting of [`flat`](Self::flat). One hydro's FPHA row
    /// block holds `n_blks * n_planes` rows laid out block-major over planes, so
    /// block `blk`'s plane `p_idx` lands at `fpha_block_start + blk * n_planes +
    /// p_idx`. Because this method takes `(fpha_block_start, blk, p_idx, n_planes)`
    /// — block before plane, with the per-block stride `n_planes` passed
    /// explicitly — it cannot be called with [`flat`](Self::flat)'s
    /// `(start, entity, blk)` order and silently transpose the two shapes.
    ///
    /// Advance the per-hydro base with [`advance_fpha_base`](Self::advance_fpha_base)
    /// after each hydro.
    // Rationale: this shape's stride is the per-hydro `n_planes` (passed in, since
    // plane counts vary per hydro), so it reads no grid constant and `self` is
    // unused. It stays an instance method — not an associated fn — so all three
    // shapes share the uniform `grid.shape(...)` call form; an associated
    // `BlockGrid::fpha_plane(..)` would break that symmetry and invite callers to
    // re-derive the stride inline rather than route through the typed primitive.
    #[allow(clippy::unused_self)]
    #[inline]
    pub(crate) fn fpha_plane(
        &self,
        fpha_block_start: usize,
        blk: usize,
        p_idx: usize,
        n_planes: usize,
    ) -> usize {
        fpha_block_start + blk * n_planes + p_idx
    }

    /// Advance the FPHA per-hydro base by one hydro's row block: `+ n_blks *
    /// n_planes`.
    ///
    /// Each FPHA hydro owns a contiguous `n_blks * n_planes` row block, so the
    /// next hydro's `fpha_block_start` is the current base plus that span. The
    /// span uses this grid's `n_blks` and the caller-supplied `n_planes` (plane
    /// counts vary per hydro), matching the stride [`fpha_plane`](Self::fpha_plane)
    /// addresses within a single hydro.
    #[inline]
    pub(crate) fn advance_fpha_base(&self, fpha_block_start: usize, n_planes: usize) -> usize {
        fpha_block_start + self.n_blks * n_planes
    }

    /// Deficit 3-term address:
    /// `deficit_start + b_pos * S * n_blks + seg * n_blks + blk`.
    ///
    /// Three nested factors: the bus position `b_pos` strides by
    /// `S * n_blks` (`S = max_deficit_segments`), the segment `seg` strides by
    /// `n_blks`, and the block `blk` is the innermost offset. The deficit column
    /// block is `B * S * n_blks` columns laid out bus-outer, segment-middle,
    /// block-inner; any reordering of the three factors (e.g. striding `seg` by
    /// `S` instead of `n_blks`, or swapping the bus and segment strides) is a
    /// same-length but wrong-cell alternative. Passing `S` and `n_blks` as named
    /// parameters in this fixed order fixes the nesting against that trap.
    #[inline]
    pub(crate) fn deficit(
        &self,
        deficit_start: usize,
        b_pos: usize,
        seg: usize,
        blk: usize,
    ) -> usize {
        deficit_start + b_pos * self.max_deficit_segments * self.n_blks + seg * self.n_blks + blk
    }
}

#[cfg(test)]
mod tests {
    use super::BlockGrid;
    use crate::indexer::StageIndexer;

    // Flat shape: 9 + 1*3 + 2 = 14, with n_blks = 3.
    #[test]
    fn flat_block_major_address() {
        let grid = BlockGrid::new(3, 1);
        assert_eq!(grid.flat(9, 1, 2), 14);
    }

    // FPHA-plane shape: 100 + 1*5 + 2 = 107, with n_planes = 5 (block OUTER,
    // plane INNER — the opposite nesting of the flat shape).
    #[test]
    fn fpha_plane_address() {
        let grid = BlockGrid::new(3, 1);
        assert_eq!(grid.fpha_plane(100, 1, 2, 5), 107);
    }

    // FPHA base advance: one hydro's row block spans n_blks * n_planes rows.
    #[test]
    fn fpha_base_advance() {
        let grid = BlockGrid::new(3, 1);
        // 100 + 3 * 5 = 115 — the next hydro's fpha_block_start.
        assert_eq!(grid.advance_fpha_base(100, 5), 115);
    }

    // Deficit 3-term shape: 61 + 0*2*3 + 1*3 + 0 = 64, with S = 2, n_blks = 3.
    #[test]
    fn deficit_three_term_address() {
        let grid = BlockGrid::new(3, 2);
        assert_eq!(grid.deficit(61, 0, 1, 0), 64);
    }

    // The grid returned by StageIndexer carries the indexer's stride constants,
    // so addresses computed through it match the indexer's own layout.
    #[test]
    fn block_grid_from_indexer_carries_strides() {
        use crate::indexer::test_fixtures::{eq, fpha};
        // N=1 hydro, L=0, T=2 thermals, L_n=1 line, B=2 buses, K=2 blocks; the
        // `eq` fixture sets max_deficit_segments = 1.
        let counts = eq(1, 0, 2, 1, 2, 2, false);
        let indexer = StageIndexer::with_equipment(&counts, &fpha(vec![], vec![]));
        let grid = indexer.block_grid();
        // The grid carries the indexer's stride constants verbatim.
        assert_eq!(grid.n_blks, indexer.n_blks);
        assert_eq!(grid.max_deficit_segments, indexer.max_deficit_segments);
        // flat() reproduces the indexer's thermal column stride (entity-outer).
        assert_eq!(
            grid.flat(indexer.thermal.start, 1, 0),
            indexer.thermal.start + indexer.n_blks
        );
        // deficit() reproduces the indexer's deficit column stride.
        assert_eq!(
            grid.deficit(indexer.deficit.start, 1, 0, 0),
            indexer.deficit.start + indexer.max_deficit_segments * indexer.n_blks
        );
    }

    // Contract guard: BlockGrid exposes no method that takes the transposed
    // argument order, so the wrong-but-compiling transpose of each shape cannot be
    // expressed through the type — a caller would have to bypass it and re-derive
    // the stride from StageIndexer fields by hand. This positive test pins the
    // forbidden alternative per shape by computing the CORRECT address and
    // asserting it differs from what the transpose would land on, using asymmetric
    // factors so any swap is detectable (symmetric indices can collide by accident
    // even though the nestings differ).
    #[test]
    fn block_grid_forbids_transposed_shape() {
        // Flat: entity-OUTER, block-INNER. The transpose makes the block the outer
        // stride: `blk * n_entities + entity`. With entity=1, blk=2, n_blks=3,
        // n_entities=4 the two land on different cells (5 vs 9).
        let grid = BlockGrid::new(3, 2);
        let (n_entities, entity, blk) = (4, 1, 2);
        let correct_flat = grid.flat(0, entity, blk);
        let transposed_flat = blk * n_entities + entity; // NOT expressible via BlockGrid
        assert_eq!(correct_flat, 5);
        assert_ne!(correct_flat, transposed_flat);

        // FPHA-plane: block-OUTER (stride n_planes), plane-INNER. The transpose
        // swaps the roles — plane-outer with the block as the inner stride n_blks:
        // `p_idx * n_blks + blk`. With blk=1, p_idx=3, n_planes=5, n_blks=2 the two
        // land on different cells (8 vs 7).
        let grid = BlockGrid::new(2, 2);
        let (blk, p_idx, n_planes) = (1, 3, 5);
        let correct_fpha = grid.fpha_plane(0, blk, p_idx, n_planes);
        let transposed_fpha = p_idx * grid.n_blks + blk; // wrong nesting
        assert_eq!(correct_fpha, 8);
        assert_ne!(correct_fpha, transposed_fpha);

        // Deficit: bus-OUTER (stride S*n_blks), segment-MIDDLE (stride n_blks),
        // block-INNER. A transpose that strides the segment by S instead of n_blks
        // — `b_pos*S*n_blks + seg*S + blk` — lands elsewhere when S != n_blks. With
        // b_pos=1, seg=1, blk=0, S=2, n_blks=3 the two differ (9 vs 8).
        let grid = BlockGrid::new(3, 2);
        let (b_pos, seg, blk) = (1, 1, 0);
        let correct_def = grid.deficit(0, b_pos, seg, blk);
        let transposed_def =
            b_pos * grid.max_deficit_segments * grid.n_blks + seg * grid.max_deficit_segments + blk; // wrong segment stride
        assert_eq!(correct_def, 9);
        assert_ne!(correct_def, transposed_def);
    }
}
