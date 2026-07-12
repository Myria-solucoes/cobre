//! The typed [`StorageBoundaryGrid`] address primitive for the chronological
//! storage-boundary family, single-owning `block_storage_col`'s three-arm
//! formula beside [`BlockGrid`](super::BlockGrid).
//!
//! Every `block_storage_col` copy (`StageLayout`, [`StageGeometry`
//! ](crate::lp_builder::StageGeometry), the generic-constraint resolver) delegates
//! to [`StorageBoundaryGrid::col`] rather than re-deriving the endpoint/interior
//! split — the wrong-but-compiling alternative is a hand-rolled copy of the match
//! whose arm order silently drifts from the others.

/// Typed storage-boundary address calculator for one SDDP stage LP.
///
/// A cheap `Copy` value carrying the two state bases and the two stage
/// constants `block_storage_col`'s three address arms need: the incoming- and
/// outgoing-state column bases (from [`StateLayout`](super::StateLayout)) and
/// the interior control-region anchor plus block count (from `StageLayout`).
#[derive(Debug, Clone, Copy, Default)]
pub struct StorageBoundaryGrid {
    storage_in_base: usize,
    storage_out_base: usize,
    storage_internal_start: usize,
    n_blks: usize,
}

impl StorageBoundaryGrid {
    /// Construct a [`StorageBoundaryGrid`] from its state bases and stride
    /// constants.
    ///
    /// Source `storage_in_base`/`storage_out_base` from the owning stage's
    /// `StateLayout` (`storage_in.start`/`storage.start`), never a hard-coded
    /// `0`, so the grid cannot disagree with a future region reorder.
    #[inline]
    #[must_use]
    pub fn new(
        storage_in_base: usize,
        storage_out_base: usize,
        storage_internal_start: usize,
        n_blks: usize,
    ) -> Self {
        Self {
            storage_in_base,
            storage_out_base,
            storage_internal_start,
            n_blks,
        }
    }

    /// Storage column at chronological boundary `k ∈ 0..=n_blks` for hydro `h`.
    /// The two endpoints are STATE columns — `k = 0 → S⁰` (incoming) and
    /// `k = n_blks → Sᴷ` (outgoing) — while `k ∈ 1..n_blks` are interior CONTROL
    /// columns (stride `n_blks − 1`, not `n_blks`).
    #[inline]
    #[must_use]
    pub fn col(&self, h: usize, k: usize) -> usize {
        match k {
            0 => self.storage_in_base + h,
            // Contract: this arm MUST precede `_` — swapped, `_` captures the
            // outgoing endpoint and addresses an interior column past the family
            // (wrong-but-compiling).
            k if k == self.n_blks => self.storage_out_base + h,
            _ => self.storage_internal_start + h * (self.n_blks - 1) + (k - 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::StorageBoundaryGrid;

    #[test]
    fn storage_boundary_grid_resolves_endpoints_and_interior() {
        let grid = StorageBoundaryGrid::new(100, 200, 50, 3);
        let h = 2;

        assert_eq!(grid.col(h, 0), 100 + h, "S⁰ endpoint");
        assert_eq!(grid.col(h, 3), 200 + h, "Sᴷ endpoint");
        assert_eq!(grid.col(h, 1), 50 + h * 2, "interior k=1");
        assert_eq!(grid.col(h, 2), 50 + h * 2 + 1, "interior k=2");
    }
}
