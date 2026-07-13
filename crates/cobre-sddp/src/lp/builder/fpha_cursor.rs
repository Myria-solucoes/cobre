use crate::hydro_models::{FphaPlane, ResolvedProductionModel};
use crate::indexer::{BlockIdx, FphaLocal, HydroSys};

use super::layout::{StageLayout, TemplateBuildCtx};

/// Walk every FPHA hyperplane row of the stage, invoking `visit` once per
/// `(FPHA hydro, block, plane)` triple with the resolved plane and its LP row.
///
/// The single owner of the FPHA row-cursor arithmetic: both the bounds fill
/// ([`super::rows::fill_fpha_rows`]) and the coefficient fill
/// ([`super::entries::fill_fpha_entries`]) drive off this walker, so a one-sided
/// edit that lands the bounds and the coefficients on different rows is impossible.
///
/// The per-hydro block start advances by the cumulative `n_blks * n_planes` prefix
/// sum over preceding FPHA hydros — REQUIRED because plane counts vary per hydro: a
/// uniform `local_idx * n_blks * n_planes` stride would overlap a later,
/// fewer-plane hydro onto an earlier hydro's rows. Matches `FphaRowRange::start`.
///
/// The closure is a monomorphised `FnMut` borrowing its target buffer (no
/// `Box<dyn>`, no intermediate `Vec`), so the build allocates nothing and the
/// byte-identical `(row, value)` push order is preserved.
pub(super) fn for_each_fpha_plane<F>(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    mut visit: F,
) where
    F: FnMut(FphaLocal, HydroSys, BlockIdx, usize, &FphaPlane, usize),
{
    let n_blks = layout.n_blks;
    let grid = layout.block_grid();
    let mut fpha_block_start = layout.row_fpha_start();
    for (local_idx, &h) in layout.fpha_hydro_indices.iter().enumerate() {
        let planes = match ctx.production_models.model(h.get(), stage_idx) {
            ResolvedProductionModel::Fpha { planes, .. } => planes,
            ResolvedProductionModel::ConstantProductivity { .. } => {
                debug_assert!(
                    false,
                    "fpha_hydro_indices contains hydro {} but model is ConstantProductivity",
                    h.get()
                );
                continue;
            }
        };
        let n_planes = planes.len();
        debug_assert_eq!(
            n_planes,
            layout.fpha_planes_per_hydro[local_idx],
            "plane count mismatch for FPHA hydro {} at stage {stage_idx}",
            h.get()
        );
        for blk in (0..n_blks).map(BlockIdx::new) {
            for (p_idx, plane) in planes.iter().enumerate() {
                // Block OUTER (stride n_planes), plane INNER — the opposite nesting
                // of the flat shape; the distinct `fpha_plane` method prevents a
                // silent transpose of the two.
                let row = grid.fpha_plane(fpha_block_start, blk, p_idx, n_planes);
                visit(FphaLocal::new(local_idx), h, blk, p_idx, plane, row);
            }
        }
        fpha_block_start = grid.advance_fpha_base(fpha_block_start, n_planes);
    }
}
