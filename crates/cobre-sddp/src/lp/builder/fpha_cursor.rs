use crate::hydro_models::{FphaPlane, ResolvedProductionModel};

use super::layout::{StageLayout, TemplateBuildCtx};

/// Walk every FPHA hyperplane row of the stage, invoking `visit` once per
/// `(FPHA hydro, block, plane)` triple with the resolved plane and its LP row.
///
/// This is the single owner of the FPHA row-cursor arithmetic. Both the bounds
/// fill ([`super::rows::fill_fpha_rows`]) and the coefficient fill
/// ([`super::entries::fill_fpha_entries`]) drive off this walker, so the cursor
/// advance and the row formula live in one place: a one-sided edit to either
/// fill is no longer possible, which would otherwise land the row bounds and the
/// matrix coefficients on different rows.
///
/// The per-hydro block start advances by the cumulative `n_blks * n_planes`
/// prefix sum over preceding FPHA hydros — REQUIRED because plane counts vary
/// per hydro, so using this hydro's plane count as a uniform stride
/// (`local_idx * n_blks * n_planes`) overlaps a later, fewer-plane hydro onto an
/// earlier hydro's rows. Within a hydro the row is
/// `block_start + blk * n_planes + p_idx`. Matches the indexer's
/// `FphaRowRange::start` ordering exactly.
///
/// The closure receives `&FphaPlane` (not a re-indexable `p_idx` alone) so each
/// fill reads exactly the coefficients it needs (`intercept` for the bounds,
/// `gamma_*` for the matrix) without a second `planes[p_idx]` lookup. The
/// closure is a monomorphised `FnMut` that borrows its target buffer — no
/// `Box<dyn>`, no intermediate `Vec` — so the build path allocates nothing and
/// the byte-identical `(row, value)` push order is preserved.
pub(super) fn for_each_fpha_plane<F>(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    mut visit: F,
) where
    F: FnMut(usize, usize, usize, usize, &FphaPlane, usize),
{
    let n_blks = layout.n_blks;
    let grid = layout.indexer.block_grid();
    let mut fpha_block_start = layout.row_fpha_start();
    for (local_idx, &h_idx) in layout.fpha_hydro_indices.iter().enumerate() {
        let planes = match ctx.production_models.model(h_idx, stage_idx) {
            ResolvedProductionModel::Fpha { planes, .. } => planes,
            ResolvedProductionModel::ConstantProductivity { .. } => {
                // Invariant: fpha_hydro_indices only ever holds FPHA hydros.
                debug_assert!(
                    false,
                    "fpha_hydro_indices contains hydro {h_idx} but model is ConstantProductivity"
                );
                continue;
            }
        };
        let n_planes = planes.len();
        debug_assert_eq!(
            n_planes, layout.fpha_planes_per_hydro[local_idx],
            "plane count mismatch for FPHA hydro {h_idx} at stage {stage_idx}"
        );
        for blk in 0..n_blks {
            for (p_idx, plane) in planes.iter().enumerate() {
                // FPHA-plane shape: block OUTER (stride n_planes), plane INNER —
                // the opposite nesting of the flat shape. Routed through the
                // distinct `fpha_plane` method so the flat `(start, entity, blk)`
                // order cannot silently transpose the two.
                let row = grid.fpha_plane(fpha_block_start, blk, p_idx, n_planes);
                visit(local_idx, h_idx, blk, p_idx, plane, row);
            }
        }
        fpha_block_start = grid.advance_fpha_base(fpha_block_start, n_planes);
    }
}
