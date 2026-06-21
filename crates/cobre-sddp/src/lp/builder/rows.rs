use cobre_core::Stage;

use crate::hydro_models::EvaporationModel;

use super::fpha_cursor::for_each_fpha_plane;
use super::layout::{StageLayout, TemplateBuildCtx};

/// Fill row lower/upper bounds for one stage.
///
/// Returns `(row_lower, row_upper)` vectors of length `layout.num_rows`.
pub(super) fn fill_stage_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
) -> (Vec<f64>, Vec<f64>) {
    let mut row_lower = vec![0.0_f64; layout.num_rows];
    let mut row_upper = vec![0.0_f64; layout.num_rows];

    fill_water_balance_rows(ctx, stage_idx, layout, &mut row_lower, &mut row_upper);
    fill_load_balance_rows(
        ctx,
        stage,
        stage_idx,
        layout,
        &mut row_lower,
        &mut row_upper,
    );
    fill_fpha_rows(ctx, stage_idx, layout, &mut row_lower, &mut row_upper);
    fill_evaporation_rows(ctx, stage_idx, layout, &mut row_lower, &mut row_upper);
    fill_operational_violation_rows(ctx, stage_idx, layout, &mut row_lower, &mut row_upper);
    fill_anticipated_fishing_rows(ctx, layout, &mut row_lower, &mut row_upper);
    fill_anticipated_state_out_def_rows(ctx, stage_idx, layout, &mut row_lower, &mut row_upper);
    fill_z_inflow_rows(ctx, stage_idx, layout, &mut row_lower, &mut row_upper);

    (row_lower, row_upper)
}

/// Fill water-balance row bounds: static RHS = ζ · (`deterministic_base_h` − `water_withdrawal_m3s_h`).
///
/// The withdrawal is a fixed schedule that reduces the effective inflow available
/// to the reservoir. Subtracting it from the base keeps the row bound correct for
/// the stage template; the PAR(p) noise innovation is added at solve time via patches.
fn fill_water_balance_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    for h_idx in 0..layout.n_h {
        let row = layout.row_water_balance_start() + h_idx;
        let base = if ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == layout.n_h {
            ctx.par_lp.deterministic_base(stage_idx, h_idx)
        } else {
            0.0
        };
        let withdrawal = ctx
            .resolved
            .bounds
            .hydro_bounds(h_idx, stage_idx)
            .water_withdrawal_m3s;
        let rhs = layout.zeta * (base - withdrawal);
        row_lower[row] = rhs;
        row_upper[row] = rhs;
    }
}

/// Fill load-balance row bounds: static RHS = `mean_mw` · `block_factor`.
///
/// Block factors from `load_factors.json` scale the mean load per block
/// (e.g., heavy/medium/light blocks). Default factor is 1.0 (no scaling).
fn fill_load_balance_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    let grid = layout.block_grid();
    for (b_idx, bus) in ctx.buses.iter().enumerate() {
        let mean_mw = ctx
            .load_models
            .iter()
            .find(|lm| lm.bus_id == bus.id && lm.stage_id == stage.id)
            .map_or(0.0, |lm| lm.mean_mw);
        for blk in 0..layout.n_blks {
            let factor = ctx
                .resolved
                .resolved_load_factors
                .factor(b_idx, stage_idx, blk);
            let row = grid.flat(layout.row_load_balance_start(), b_idx, blk);
            let rhs = mean_mw * factor;
            row_lower[row] = rhs;
            row_upper[row] = rhs;
        }
    }
}

/// Fill FPHA hyperplane row bounds: `row_lower = -INF`, `row_upper = gamma_0`
/// (the pre-scaled `intercept`).
///
/// The constraint is `g_{h,k} - gamma_v/2·v - gamma_v/2·v_in - gamma_q·q - gamma_s·s <= gamma_0`.
/// The `v`, `v_in`, `q`, `s` contributions live in the CSC matrix entries
/// ([`super::entries::fill_fpha_entries`]), so the static upper bound carries only the `intercept`.
/// Driven by [`for_each_fpha_plane`] so these bounds and the matrix coefficients
/// share one row cursor.
fn fill_fpha_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    for_each_fpha_plane(
        ctx,
        stage_idx,
        layout,
        |_local_idx, _h_idx, _blk, _p_idx, plane, row| {
            row_lower[row] = f64::NEG_INFINITY;
            row_upper[row] = plane.intercept;
        },
    );
}

/// Fill evaporation row bounds: equality `row_lower == row_upper == intercept_m3s`.
///
/// The linearised outflow is
/// `intercept_m3s + volume_slope_m3s_per_hm3/2·(v + v_in - 2·reference_volume)`.
/// The volume-dependent term `volume_slope_m3s_per_hm3/2 · v` is added via the
/// CSC matrix entry on the outgoing-storage column, so the static row bounds
/// encode only the constant `intercept_m3s`.
fn fill_evaporation_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    for (local_idx, &h_idx) in layout.evap_hydro_indices.iter().enumerate() {
        match ctx.evaporation_models.model(h_idx) {
            EvaporationModel::Linearized { coefficients, .. } => {
                debug_assert!(
                    stage_idx < coefficients.len(),
                    "stage index {stage_idx} out of bounds for evaporation coefficients (len = {})",
                    coefficients.len()
                );
                let intercept_m3s = coefficients[stage_idx].intercept_m3s;
                let row = layout.row_evap_start() + local_idx;
                row_lower[row] = intercept_m3s;
                row_upper[row] = intercept_m3s;
            }
            EvaporationModel::None => {
                // Should never happen: evap_hydro_indices only contains linearized hydros.
                // No row is written; release builds skip this hydro.
                debug_assert!(
                    false,
                    "evap_hydro_indices contains hydro {h_idx} but model is None"
                );
            }
        }
    }
}

/// Fill z-inflow definition row bounds: equality with RHS = `base_h` (m3/s).
///
/// The base is the deterministic PAR base inflow (before noise), NOT multiplied
/// by ζ and NOT reduced by withdrawal. The noise component (sigma · eta) is added
/// at solve time via `PatchBuffer` Category 5.
fn fill_z_inflow_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    for h_idx in 0..layout.n_h {
        let row = layout.row_z_inflow_start() + h_idx;
        let base = if ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == layout.n_h {
            ctx.par_lp.deterministic_base(stage_idx, h_idx)
        } else {
            0.0
        };
        row_lower[row] = base;
        row_upper[row] = base;
    }
}

/// Fill row bounds for the 4 operational violation constraint families.
///
/// Per-block formulation: one row per hydro per block. RHS is in rate units
/// (m3/s for flow families, MW for generation).
///
/// - **Min outflow** (`>=`): `row_lower = min_outflow_m3s`, `row_upper = +INF`.
/// - **Max outflow** (`<=`): `row_lower = -INF`, `row_upper = max_outflow_m3s`
///   (or `+INF` when the bound is absent, making the row non-binding).
/// - **Min turbine** (`>=`): `row_lower = min_turbined_m3s`, `row_upper = +INF`.
/// - **Min generation** (`>=`): `row_lower = min_generation_mw`, `row_upper = +INF`.
fn fill_operational_violation_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    // All four operational-violation row families share the same per-hydro bound
    // lookup, so the bound is read once per hydro and a `(row_start, lower, upper)`
    // descriptor is built from it. Each family targets `row_lower`/`row_upper` by its
    // own computed row index, so the visit order is irrelevant to the result. The
    // descriptor order is nonetheless pinned to the canonical row-region order
    // (min-outflow, max-outflow, min-turbine, min-generation) so the source-level
    // write order stays auditable against the layout. Per-family sense:
    //   min-outflow   (>=): LHS + sigma >= min_outflow_m3s
    //   max-outflow   (<=): LHS - sigma <= max_outflow_m3s
    //   min-turbine   (>=): LHS + sigma >= min_turbined_m3s
    //   min-generation(>=): LHS + sigma >= min_generation_mw
    let grid = layout.block_grid();
    for h_idx in 0..layout.n_h {
        let hb = ctx.resolved.bounds.hydro_bounds(h_idx, stage_idx);
        let families = [
            (
                layout.row_min_outflow_start(),
                hb.min_outflow_m3s,
                f64::INFINITY,
            ),
            (
                layout.row_max_outflow_start(),
                f64::NEG_INFINITY,
                hb.max_outflow_m3s.unwrap_or(f64::INFINITY),
            ),
            (
                layout.row_min_turbine_start(),
                hb.min_turbined_m3s,
                f64::INFINITY,
            ),
            (
                layout.row_min_generation_start(),
                hb.min_generation_mw,
                f64::INFINITY,
            ),
        ];
        for (row_start, lower, upper) in families {
            for blk in 0..layout.n_blks {
                let row = grid.flat(row_start, h_idx, blk);
                row_lower[row] = lower;
                row_upper[row] = upper;
            }
        }
    }
}

/// Fill row bounds for anticipated-fishing equality constraints.
///
/// For each anticipated plant `i`, sets one row to equality `0 == 0`. The
/// fishing constraint balances per-block thermal generation (`MWh`) against the
/// committed power level in the `anticipated_state` slot
/// (`MW` × `block_hours_total` = `MWh`).
///
/// The predicate is always-active: one row per anticipated plant at every stage.
/// When `n_anticipated == 0`, this function is a no-op.
pub(super) fn fill_anticipated_fishing_rows(
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    // Every anticipated plant emits a fishing row at the dense offset
    // `row_anticipated_fishing_start + local_idx`. The offset is dense — not a
    // sparse `active_pos` offset — precisely because the fishing constraint is
    // always active for every anticipated plant.
    for local_idx in 0..ctx.n_anticipated {
        let row = layout.anticipated.row_anticipated_fishing_start + local_idx;
        row_lower[row] = 0.0;
        row_upper[row] = 0.0;
    }
    debug_assert_eq!(
        ctx.n_anticipated, layout.anticipated.n_anticipated_fishing_rows,
        "fill_anticipated_fishing_rows: row count must equal n_anticipated"
    );
}

/// Fill row bounds for the `anticipated_state_out` definition equality rows.
///
/// For each active anticipated plant (`stage_idx + K_i < n_stages`),
/// sets one row to equality `0 == 0`. Inactive plants emit no row, so rows are
/// packed at the sparse `active_pos` offset. This gates on
/// `stage_idx + K_i < n_stages`, unlike [`fill_anticipated_fishing_rows`], which
/// is always active and emits one row per plant at a dense offset.
///
/// No-op when `n_anticipated == 0` or when no plant is active at
/// `stage_idx`.
pub(super) fn fill_anticipated_state_out_def_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    let n_stages = ctx.resolved.bounds.n_stages();
    let mut active_pos: usize = 0;
    for local_idx in 0..ctx.n_anticipated {
        if !layout.is_anticipated_decision_active(local_idx, stage_idx, n_stages) {
            continue;
        }
        let row = layout.anticipated.row_anticipated_state_out_def_start + active_pos;
        row_lower[row] = 0.0;
        row_upper[row] = 0.0;
        active_pos += 1;
    }
    debug_assert_eq!(
        active_pos, layout.anticipated.n_anticipated_state_out_def_rows,
        "fill_anticipated_state_out_def_rows: active_pos mismatch at stage {stage_idx}"
    );
}
