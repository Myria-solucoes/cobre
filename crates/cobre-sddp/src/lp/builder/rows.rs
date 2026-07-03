use cobre_core::{BlockMode, Stage};

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

    fill_water_balance_rows(
        ctx,
        stage,
        stage_idx,
        layout,
        &mut row_lower,
        &mut row_upper,
    );
    fill_bucket_definition_rows(layout, &mut row_lower, &mut row_upper);
    fill_filling_target_rows(ctx, stage.id, layout, &mut row_lower, &mut row_upper);
    fill_filled_min_storage_floor_rows(ctx, stage_idx, layout, &mut row_lower, &mut row_upper);
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

/// Fill water-balance row bounds: static RHS = ζ · (`deterministic_base_h` −
/// `water_withdrawal_m3s_h`); the PAR(p) noise innovation is added at solve time.
///
/// A `PreFilling` hydro's row is the frozen identity `v_h − v_h_in = 0` (matrix
/// entries by [`super::entries::fill_state_and_water_entries`]), so its RHS is `0`,
/// NOT `ζ·(base − withdrawal)`: its base inflow rides the routed `z_h` column on
/// the short-circuit target row, and its withdrawal DEMAND transfers to that
/// target's RHS below. The solve-time noise patch is neutralized by zeroing the
/// hydro's `noise_scale`, so the `0` RHS survives; a nonzero RHS here would break
/// the frozen identity even with the noise patch zeroed.
///
/// In `BlockMode::Chronological` the single per-hydro RHS splits into `K` per-block
/// row bounds `τ_k·(base − withdrawal)` (block-major, mirroring the entries side);
/// summing them recovers the parallel `ζ·(base − withdrawal)` since `Σ_k τ_k = ζ`.
fn fill_water_balance_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    match stage.block_mode {
        BlockMode::Parallel => {
            fill_parallel_water_rows(ctx, stage, stage_idx, layout, row_lower, row_upper);
        }
        BlockMode::Chronological => {
            fill_chronological_water_rows(ctx, stage, stage_idx, layout, row_lower, row_upper);
        }
    }
}

fn fill_parallel_water_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    let has_par = ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == layout.n_h;
    for h_idx in 0..layout.n_h {
        let row = layout.row_water_balance_start() + h_idx;
        if super::entries::is_prefilling(ctx, stage, h_idx) {
            // Frozen identity: RHS 0 (the base inflow rides the routed `z_h` column).
            row_lower[row] = 0.0;
            row_upper[row] = 0.0;
            continue;
        }
        let base = if has_par {
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

    // Transfer each PreFilling hydro's withdrawal DEMAND (`ζ·withdrawal_h`) to its
    // short-circuit target `d`'s RHS, absorbed by `d`'s withdrawal slacks. `d` MUST
    // be the SAME target `fill_prefilling_shortcircuit` routed the matrix terms to,
    // so the RHS transfer and the matrix coupling agree; routing onto the immediate
    // `downstream(h)` would land it on a PreFilling downstream's frozen-identity RHS
    // (which must stay `0`). A second pass, since `d` may be filled before or after
    // `h` in index order; sink case transfers nothing.
    for h_idx in 0..layout.n_h {
        if !super::entries::is_prefilling(ctx, stage, h_idx) {
            continue;
        }
        let Some(d_idx) = super::entries::resolve_shortcircuit_target(ctx, stage, h_idx) else {
            continue;
        };
        let withdrawal_h = ctx
            .resolved
            .bounds
            .hydro_bounds(h_idx, stage_idx)
            .water_withdrawal_m3s;
        let delta = layout.zeta * withdrawal_h;
        let row_d = layout.row_water_balance_start() + d_idx;
        row_lower[row_d] -= delta;
        row_upper[row_d] -= delta;
    }
}

/// Per-block water-balance RHS for chronological mode: each Operating/Filling hydro
/// gets `K` rows `τ_k·(base − withdrawal)` (block-major `row_water + h·K + (k−1)`),
/// with `τ_k` replacing `ζ` so `Σ_k` recovers the parallel total. A `PreFilling`
/// hydro gets `K` frozen-identity rows with RHS `0` (block-major), and its
/// withdrawal transfers per block (`−τ_k·withdrawal_h`) to the short-circuit
/// target's block rows, mirroring the entries side.
fn fill_chronological_water_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    let n_blks = layout.n_blks;
    let has_par = ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == layout.n_h;
    for h_idx in 0..layout.n_h {
        if super::entries::is_prefilling(ctx, stage, h_idx) {
            for k in 1..=n_blks {
                let row = layout.row_water_balance_start() + h_idx * n_blks + (k - 1);
                row_lower[row] = 0.0;
                row_upper[row] = 0.0;
            }
            continue;
        }
        let base = if has_par {
            ctx.par_lp.deterministic_base(stage_idx, h_idx)
        } else {
            0.0
        };
        let withdrawal = ctx
            .resolved
            .bounds
            .hydro_bounds(h_idx, stage_idx)
            .water_withdrawal_m3s;
        for k in 1..=n_blks {
            let blk = k - 1;
            let row = layout.row_water_balance_start() + h_idx * n_blks + blk;
            let tau_k = stage.blocks[blk].duration_hours * super::M3S_TO_HM3;
            let rhs = tau_k * (base - withdrawal);
            row_lower[row] = rhs;
            row_upper[row] = rhs;
        }
    }

    for h_idx in 0..layout.n_h {
        if !super::entries::is_prefilling(ctx, stage, h_idx) {
            continue;
        }
        let Some(d_idx) = super::entries::resolve_shortcircuit_target(ctx, stage, h_idx) else {
            continue;
        };
        let withdrawal_h = ctx
            .resolved
            .bounds
            .hydro_bounds(h_idx, stage_idx)
            .water_withdrawal_m3s;
        for k in 1..=n_blks {
            let blk = k - 1;
            let tau_k = stage.blocks[blk].duration_hours * super::M3S_TO_HM3;
            let row_d = layout.row_water_balance_start() + d_idx * n_blks + blk;
            row_lower[row_d] -= tau_k * withdrawal_h;
            row_upper[row_d] -= tau_k * withdrawal_h;
        }
    }
}

/// Fill the travel-time bucket-definition equality row bounds (`0 == 0`), one
/// row per declared `(plant, lag)` bucket. Dense and ALWAYS active regardless
/// of stage (mirrors [`fill_anticipated_fishing_rows`]'s always-active dense
/// offset, not the sparse `active_pos` offset
/// [`fill_anticipated_state_out_def_rows`] uses) — every declared bucket keeps
/// a definition row at every stage. Empty when `layout.state.b_total == 0`.
fn fill_bucket_definition_rows(layout: &StageLayout, row_lower: &mut [f64], row_upper: &mut [f64]) {
    let row_start = layout.row_bucket_definition_start();
    for b in 0..layout.state.b_total {
        let row = row_start + b;
        row_lower[row] = 0.0;
        row_upper[row] = 0.0;
    }
}

/// Fill the soft filling-target row bounds (`v_h + σ_fill ≥ V_target[t]`, in hm³):
/// `row_lower = V_target[t]`, `row_upper = +∞`. LHS coefficients are emitted by
/// [`super::entries::fill_filling_target_entries`].
///
/// **Contract — the RHS is the per-stage `V_target[t]` (backward-anchored), NOT
/// `min_storage` at every stage.** `V_target[t]` is the precomputed trajectory
/// folded backward from the dead volume in
/// [`build_filling_v_target`](super::template::build_filling_v_target), so only the
/// LAST Filling stage's floor equals `min_storage_hm3` and earlier floors are
/// strictly lower. Writing `min_storage_hm3` at every Filling stage would demand
/// the full dead volume from the FIRST Filling stage — an over-strict floor the
/// soft slack absorbs at cost every stage. A per-stage helper cannot see other
/// stages' ζ·rate, so the trajectory MUST come from the precompute.
///
/// SOFT `≥` (the `σ_fill` slack relaxes it), never a hard column bound on `v_h`, so
/// a hydro that fills short keeps a feasible LP.
fn fill_filling_target_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage_id: i32,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    let row_start = layout.row_filling_target_start();
    for (local_idx, &h_idx) in layout.filling_target_hydro_indices.iter().enumerate() {
        // Fail loud on a lookup miss rather than writing a `0.0` floor, which would
        // make `v + σ_fill ≥ 0` trivially true and silently neutralize the
        // constraint. Membership and the `build_filling_v_target` precompute share
        // the Filling-phase predicate, so a miss is a construction bug between them.
        let Some(v_target) = ctx.filling_v_target.get(&(h_idx, stage_id)).copied() else {
            unreachable!(
                "no V_target for filling hydro {h_idx} at stage id {stage_id}: \
                 filling_target membership and the V_target precompute disagree"
            );
        };
        let row = row_start + local_idx;
        row_lower[row] = v_target;
        row_upper[row] = f64::INFINITY;
    }
}

/// Fill the soft operating-floor row bounds (`v_h + σ^{v-} ≥ min_storage_hm3`, in
/// hm³): `row_lower = min_storage_hm3`, `row_upper = +∞`. LHS coefficients are
/// emitted by [`super::entries::fill_filled_min_storage_floor_entries`].
///
/// `min_storage_hm3` is the RESOLVED per-stage dead volume
/// (`hydro_bounds(h_idx, stage_idx).min_storage_hm3`); reading the raw
/// `Hydro.min_storage_hm3` entity field instead would drop any per-stage override.
///
/// SOFT `≥`, never a hard column bound: `fill_storage_columns` relaxes a filling
/// hydro's hard floor to `0` so a reservoir that finished filling short keeps a
/// feasible LP, with `storage_violation_below_cost` driving `σ^{v-} → 0`.
///
/// DISTINCT from `fill_filling_target_rows` (`σ_fill`): same `≥` shape but a
/// non-overlapping stage scope (Operating vs Filling), a different RHS
/// (`min_storage` here vs the per-stage `V_target[t]` there), and a different slack
/// column and cost.
fn fill_filled_min_storage_floor_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    let row_start = layout.row_filled_min_storage_floor_start();
    for (local_idx, &h_idx) in layout
        .filled_min_storage_floor_hydro_indices
        .iter()
        .enumerate()
    {
        let min_storage = ctx
            .resolved
            .bounds
            .hydro_bounds(h_idx, stage_idx)
            .min_storage_hm3;
        let row = row_start + local_idx;
        row_lower[row] = min_storage;
        row_upper[row] = f64::INFINITY;
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

/// Fill FPHA hyperplane row bounds: `row_lower = -INF`, `row_upper = gamma_0` (the
/// pre-scaled `intercept`). The `v`/`v_in`/`q`/`s` contributions live in the matrix
/// entries ([`super::entries::fill_fpha_entries`]), so the upper bound carries only
/// the intercept. Driven by [`for_each_fpha_plane`] so these bounds and the matrix
/// coefficients share one row cursor.
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

/// Fill evaporation row bounds: equality `row_lower == row_upper == intercept_m3s`,
/// one row per `(evap hydro, block)` (block-major `row_evap_start + local * n_blks +
/// blk`, in lockstep with the entries side). The volume-dependent term lives in the
/// matrix entries ([`super::entries::fill_evaporation_entries`]), so the row bounds
/// encode only the constant intercept, replicated across the hydro's `K` block rows.
fn fill_evaporation_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    let n_blks = layout.n_blks;
    for (local_idx, &h_idx) in layout.evap_hydro_indices.iter().enumerate() {
        match ctx.evaporation_models.model(h_idx) {
            EvaporationModel::Linearized { coefficients, .. } => {
                debug_assert!(
                    stage_idx < coefficients.len(),
                    "stage index {stage_idx} out of bounds for evaporation coefficients (len = {})",
                    coefficients.len()
                );
                let intercept_m3s = coefficients[stage_idx].intercept_m3s;
                for blk in 0..n_blks {
                    let row = layout.row_evap_start() + local_idx * n_blks + blk;
                    row_lower[row] = intercept_m3s;
                    row_upper[row] = intercept_m3s;
                }
            }
            EvaporationModel::None => {
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
/// at solve time via [`PatchBuffer::fill_z_inflow_patches`].
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
    // Each family writes its own computed row index, so the visit order does not
    // affect the result; the descriptor order is nonetheless pinned to the canonical
    // row-region order (min-outflow, max-outflow, min-turbine, min-generation) so the
    // write order stays auditable against the layout.
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

/// Fill anticipated-fishing equality row bounds: `0 == 0` per anticipated plant.
/// Always-active: one row per plant at every stage.
pub(super) fn fill_anticipated_fishing_rows(
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    // Dense offset (`+ local_idx`), not a sparse `active_pos` offset, because the
    // fishing constraint is always active for every anticipated plant.
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

/// Fill the `anticipated_state_out` definition equality row bounds (`0 == 0`) for
/// each active plant (`stage_idx + K_i < n_stages`). Inactive plants emit no row,
/// so rows pack at the SPARSE `active_pos` offset — unlike
/// [`fill_anticipated_fishing_rows`], which is always active and uses a dense offset.
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
        if !layout.is_anticipated_decision_active(
            local_idx,
            stage_idx,
            n_stages,
            &ctx.anticipated_windows,
            &ctx.study_stage_ids,
        ) {
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
