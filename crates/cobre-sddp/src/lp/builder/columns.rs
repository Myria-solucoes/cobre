use cobre_core::Stage;

use crate::hydro_models::{EvaporationModel, ResolvedProductionModel};

use super::EVAPORATION_FLOW_SAFETY_MARGIN;
use super::layout::{StageLayout, TemplateBuildCtx};

/// Mutable column-bound and objective buffers shared by all fill helpers.
///
/// Passed by mutable reference to each `fill_*_columns` helper so that the
/// orchestrator `fill_stage_columns` call sites stay on a single line each.
/// This is an analogue of `LpMatrixBuffers` for the column-fill path.
pub(super) struct ColumnBufs<'a> {
    pub(super) col_lower: &'a mut [f64],
    pub(super) col_upper: &'a mut [f64],
    pub(super) objective: &'a mut [f64],
}

/// Fill column lower/upper bounds and objective coefficients for one stage.
///
/// Returns `(col_lower, col_upper, objective)` vectors of length `layout.num_cols`.
pub(super) fn fill_stage_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut col_lower = vec![0.0_f64; layout.num_cols];
    let mut col_upper = vec![f64::INFINITY; layout.num_cols];
    let mut objective = vec![0.0_f64; layout.num_cols];
    let total_stage_hours: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
    let b = &mut ColumnBufs {
        col_lower: &mut col_lower,
        col_upper: &mut col_upper,
        objective: &mut objective,
    };

    fill_storage_columns(ctx, stage_idx, layout, b);
    fill_ar_lag_columns(layout, b);
    fill_anticipated_state_columns(layout, b);
    fill_theta_column(layout, b);
    fill_turbine_columns(ctx, stage, stage_idx, layout, b);
    fill_spillage_columns(ctx, stage, stage_idx, layout, b);
    fill_diversion_columns(ctx, stage, stage_idx, layout, b);
    fill_thermal_columns(ctx, stage, stage_idx, layout, b);
    fill_anticipated_decision_columns(ctx, stage_idx, layout, b);
    fill_anticipated_state_out_columns(ctx, stage_idx, layout, b);
    fill_anticipated_decision_objective(ctx, stage_idx, layout, b);
    zero_anticipated_delivery_thermal_cost(ctx, layout, b);
    fill_line_columns(ctx, stage, stage_idx, layout, b);
    fill_deficit_and_excess_columns(ctx, stage, stage_idx, layout, b);
    fill_inflow_slack_columns(ctx, stage_idx, layout, total_stage_hours, b);
    fill_fpha_generation_columns(ctx, stage_idx, layout, b);
    fill_evaporation_columns(ctx, stage_idx, layout, total_stage_hours, b);
    fill_withdrawal_slack_columns(ctx, stage_idx, layout, total_stage_hours, b);
    fill_operational_slack_columns(ctx, stage, stage_idx, layout, b);
    fill_ncs_columns(ctx, stage, stage_idx, layout, b);
    fill_pumping_columns(ctx, stage_idx, layout, b);
    fill_z_inflow_columns(layout, b);

    (col_lower, col_upper, objective)
}

/// Outgoing and incoming storage columns.
///
/// Outgoing storage `v_h` gets stage-specific bounds `[min_storage, max_storage]`.
/// Incoming storage `v_in_h` is unconstrained (fixed at solve-time by the
/// storage-fixing equality row).
fn fill_storage_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let hb = ctx.resolved.bounds.hydro_bounds(h_idx, stage_idx);
        bufs.col_lower[h_idx] = hb.min_storage_hm3;
        bufs.col_upper[h_idx] = hb.max_storage_hm3;
        bufs.col_lower[layout.col_storage_in_start() + h_idx] = f64::NEG_INFINITY;
        bufs.col_upper[layout.col_storage_in_start() + h_idx] = f64::INFINITY;
    }
}

/// AR lag columns: unconstrained (signed).
fn fill_ar_lag_columns(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    let n_lag_cols = layout.lag_order * layout.n_h;
    for lag_col in layout.col_inflow_lags_start()..layout.col_inflow_lags_start() + n_lag_cols {
        bufs.col_lower[lag_col] = f64::NEG_INFINITY;
        bufs.col_upper[lag_col] = f64::INFINITY;
    }
}

/// Anticipated-state columns: intentionally unconstrained bounds `(-INF, +INF)`.
///
/// Writes `(-INF, +INF)` on every `n_ant_state` anticipated-state columns.
/// The columns are stored in slot-major, plant-minor order:
/// `col = col_anticipated_state_start + slot * n_anticipated + plant`.
/// Bounds are left open because the binding constraint comes from the
/// `n_ant_state` state-fixing equality rows whose RHS values are patched
/// at solve time by `fill_state_patches`. Mirror of `fill_ar_lag_columns`.
///
/// No-op when `n_anticipated == 0` (`n_ant_state == 0`).
fn fill_anticipated_state_columns(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    for slot in 0..layout.k_max {
        for plant in 0..layout.n_anticipated {
            let col = layout.col_anticipated_state_start() + slot * layout.n_anticipated + plant;
            bufs.col_lower[col] = f64::NEG_INFINITY;
            bufs.col_upper[col] = f64::INFINITY;
        }
    }
}

/// Fill column bounds for the `anticipated_state_out` columns.
///
/// For each anticipated plant `i`:
/// - Active (`stage_idx + K_i < n_stages`): bounds `[-INF, +INF]`.
///   The `state_out` value is pinned to `decision_col[i]` by the
///   `anticipated_state_out_def` equality row (filled by
///   `fill_anticipated_state_out_def_entries`).
/// - Inactive (`stage_idx + K_i >= n_stages`): bounds `[0, 0]`. The
///   presolver eliminates the column. The definition row is NOT emitted
///   for inactive plants (lockstep invariant: zero-bound iff no def row).
///
/// Objective coefficient stays at 0.0 (vec default) — the `state_out` column
/// carries no direct cost; the cost flows through the cut machinery.
///
/// No-op when `n_anticipated == 0` (loop iterates zero times).
pub(super) fn fill_anticipated_state_out_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    let n_stages = ctx.resolved.bounds.n_stages();
    for local_idx in 0..ctx.n_anticipated {
        let col = layout.anticipated.col_anticipated_state_out_start + local_idx;
        if layout
            .indexer
            .is_anticipated_decision_active(local_idx, stage_idx, n_stages)
        {
            bufs.col_lower[col] = f64::NEG_INFINITY;
            bufs.col_upper[col] = f64::INFINITY;
        } else {
            bufs.col_lower[col] = 0.0;
            bufs.col_upper[col] = 0.0;
        }
    }
    let active_count = (0..ctx.n_anticipated)
        .filter(|&i| {
            layout
                .indexer
                .is_anticipated_decision_active(i, stage_idx, n_stages)
        })
        .count();
    debug_assert_eq!(
        active_count, layout.anticipated.n_anticipated_state_out_def_rows,
        "active state_out column count must match def-row count at stage {stage_idx}"
    );
}

/// Theta column: bounded below by zero so iteration-1 LPs with empty cut pools
/// are bounded rather than unbounded.
fn fill_theta_column(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    bufs.col_lower[layout.col_theta()] = 0.0;
    bufs.col_upper[layout.col_theta()] = f64::INFINITY;
    bufs.objective[layout.col_theta()] = 1.0;
}

/// Turbine columns per hydro per block.
///
/// For constant-productivity hydros, caps turbine flow so that
/// `productivity * turbined <= max_generation_mw` (derated capacity).
/// Carries `turbined_cost * block_hours` in the objective on every hydro's
/// turbine column regardless of production model — the turbined cost
/// applies to every plant, not only FPHA hydros.
fn fill_turbine_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let hb = ctx.resolved.bounds.hydro_bounds(h_idx, stage_idx);
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        let model = ctx.production_models.model(h_idx, stage_idx);
        let turb_upper = match model {
            ResolvedProductionModel::ConstantProductivity { productivity }
                if *productivity > 0.0 =>
            {
                hb.max_turbined_m3s.min(hb.max_generation_mw / productivity)
            }
            _ => hb.max_turbined_m3s,
        };
        for blk in 0..layout.n_blks {
            let col = layout.turbine_col(h_idx, blk);
            bufs.col_lower[col] = 0.0;
            bufs.col_upper[col] = turb_upper;
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col] = hp.turbined_cost * block_hours;
        }
    }
}

/// Spillage columns per hydro per block.
fn fill_spillage_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.spillage_col(h_idx, blk);
            bufs.col_upper[col] = f64::INFINITY;
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col] = hp.spillage_cost * block_hours;
        }
    }
}

/// Diversion columns per hydro per block.
///
/// Dense allocation: all hydros get columns; those without diversion have
/// bounds `[0, 0]` and are eliminated by presolve.
fn fill_diversion_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (h_idx, _hydro) in ctx.hydros.iter().enumerate() {
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        // Contract: the diversion upper bound is the per-stage RESOLVED value
        // (`hydro_bounds.parquet` override into `HydroStageBounds.max_diversion_m3s`),
        // NOT the declaration-time `hydro.diversion.max_flow_m3s` — reading the entity
        // directly silently drops any wired per-stage override. Mirror every sibling
        // column family (e.g. `fill_thermal_columns`).
        let max_div = ctx
            .resolved
            .bounds
            .hydro_bounds(h_idx, stage_idx)
            .max_diversion_m3s
            .unwrap_or(0.0);
        for blk in 0..layout.n_blks {
            let col = layout.diversion_col(h_idx, blk);
            bufs.col_lower[col] = 0.0;
            bufs.col_upper[col] = max_div;
            if max_div > 0.0 {
                let block_hours = stage.blocks[blk].duration_hours;
                bufs.objective[col] = hp.diversion_cost * block_hours;
            }
        }
    }
}

/// Thermal columns per thermal per block.
fn fill_thermal_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (t_idx, _thermal) in ctx.thermals.iter().enumerate() {
        let tb = ctx.resolved.bounds.thermal_bounds(t_idx, stage_idx);
        let marginal_cost_per_mwh = tb.cost_per_mwh;
        for blk in 0..layout.n_blks {
            let col = layout
                .indexer
                .block_grid()
                .flat(layout.col_thermal_start(), t_idx, blk);
            bufs.col_lower[col] = tb.min_generation_mw;
            bufs.col_upper[col] = tb.max_generation_mw;
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col] = marginal_cost_per_mwh * block_hours;
        }
    }
}

/// Anticipated-decision columns: one per anticipated thermal, stage-level.
///
/// For each anticipated plant `i` at decision stage `t`:
/// - If `t + K_i < n_stages` (active): bounds come from `thermal_bounds(thermal_idx, t + K_i)`,
///   the delivery-stage bounds.
/// - Else (inactive, `t + K_i >= n_stages`): bounds are `[0, 0]`; the presolver eliminates.
///
/// The boundary case `t + K_i == n_stages` is excluded: the delivery stage
/// would fall outside the study horizon `[0, n_stages)`, so no delivery LP is
/// built for it. Pricing such a commitment with no enforcement would create a
/// cost-only column with no physical delivery.
///
/// Objective coefficients are NOT set here; they are filled by a separate pass.
fn fill_anticipated_decision_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    let n_stages = ctx.resolved.bounds.n_stages();
    for local_idx in 0..ctx.n_anticipated {
        let k_i = ctx.anticipated_lead_stages[local_idx];
        let thermal_idx = ctx.anticipated_thermal_indices[local_idx];
        let col = layout.anticipated.col_anticipated_decision_start + local_idx;
        if layout
            .indexer
            .is_anticipated_decision_active(local_idx, stage_idx, n_stages)
        {
            let delivery_stage = stage_idx + k_i;
            let tb = ctx
                .resolved
                .bounds
                .thermal_bounds(thermal_idx, delivery_stage);
            bufs.col_lower[col] = tb.min_generation_mw;
            bufs.col_upper[col] = tb.max_generation_mw;
        } else {
            // Inactive: bounds [0, 0]; the presolver will eliminate.
            bufs.col_lower[col] = 0.0;
            bufs.col_upper[col] = 0.0;
        }
        // Objective coefficient left at default 0.0; set by fill_anticipated_decision_objective.
    }
}

/// Set NPV-discounted objective coefficients for anticipated-decision columns.
///
/// For each anticipated plant `i` active at decision stage `t`
/// (i.e. `t + K_i < n_stages`), writes:
///
/// ```text
/// objective[col_anticipated_decision_start + i] =
///     cost_per_mwh(thermal_idx, delivery_stage)
///     * total_hours_per_stage[delivery_stage]
///     * cumulative_discount_factors[delivery_stage]
/// ```
///
/// This encodes the present-value cost of committing one MW at stage `t`
/// for delivery at stage `t + K_i`. Inactive plants leave objective at `0.0`
/// (the vec initialisation default); their `[0, 0]` bounds make the LP effect
/// identical to not having the column.
///
/// The written coefficient is UNSCALED: the caller (`build_single_stage_template`)
/// divides every non-theta objective entry by `COST_SCALE_FACTOR` after this
/// function returns.
fn fill_anticipated_decision_objective(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    let n_stages = ctx.resolved.bounds.n_stages();
    for local_idx in 0..ctx.n_anticipated {
        let k_i = ctx.anticipated_lead_stages[local_idx];
        let col = layout.anticipated.col_anticipated_decision_start + local_idx;
        if layout
            .indexer
            .is_anticipated_decision_active(local_idx, stage_idx, n_stages)
        {
            let delivery_stage = stage_idx + k_i;
            let thermal_idx = ctx.anticipated_thermal_indices[local_idx];
            let tb = ctx
                .resolved
                .bounds
                .thermal_bounds(thermal_idx, delivery_stage);
            let delivery_hours = ctx.total_hours_per_stage[delivery_stage];
            let d_factor = ctx.cumulative_discount_factors[delivery_stage];
            bufs.objective[col] = tb.cost_per_mwh * delivery_hours * d_factor;
        }
        // Inactive plants: objective stays at 0.0 (the vec default).
        // The bounds [0, 0] from fill_anticipated_decision_columns ensure the
        // column has no LP effect regardless of objective value.
    }
}

/// Zero out per-block thermal objective coefficients for every anticipated plant
/// at every stage where its column exists.
///
/// The anticipated-fishing constraint is always active, so the per-block
/// generation cost is zeroed unconditionally for every anticipated plant `i` at
/// every stage. This prevents double-counting: the generation is already priced
/// at the decision stage via `fill_anticipated_decision_objective`; the LP must
/// consume it at zero marginal cost.
///
/// Must be called AFTER `fill_thermal_columns`, which writes the standard
/// non-zero cost for all thermals at all stages. This function overwrites that
/// cost for every anticipated thermal at every stage.
///
/// The assignment `= 0.0` (not `*= 0.0`) is deliberate: a clean write that
/// survives floating-point anomalies. Dividing zero by `COST_SCALE_FACTOR`
/// (the post-loop in `build_single_stage_template`) still gives zero.
pub(super) fn zero_anticipated_delivery_thermal_cost(
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    let grid = layout.indexer.block_grid();
    for local_idx in 0..ctx.n_anticipated {
        let thermal_idx = ctx.anticipated_thermal_indices[local_idx];
        for blk in 0..layout.n_blks {
            let col = grid.flat(layout.col_thermal_start(), thermal_idx, blk);
            bufs.objective[col] = 0.0;
        }
    }
}

/// Line columns per line per block (forward and reverse).
///
/// Exchange factors from `exchange_factors.json` scale the stage-level
/// capacity bounds per block. Default factor is `(1.0, 1.0)` (no scaling).
fn fill_line_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (l_idx, _line) in ctx.lines.iter().enumerate() {
        let lb = ctx.resolved.bounds.line_bounds(l_idx, stage_idx);
        let lp = ctx.resolved.penalties.line_penalties(l_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let (df, rf) = ctx
                .resolved
                .resolved_exchange_factors
                .factors(l_idx, stage_idx, blk);
            let col_fwd = layout.line_fwd_col(l_idx, blk);
            let col_rev = layout.line_rev_col(l_idx, blk);
            bufs.col_upper[col_fwd] = lb.direct_mw * df;
            bufs.col_upper[col_rev] = lb.reverse_mw * rf;
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col_fwd] = lp.exchange_cost * block_hours;
            bufs.objective[col_rev] = lp.exchange_cost * block_hours;
        }
    }
}

/// Deficit and excess columns per bus per block.
///
/// The deficit region uses a uniform stride of `max_deficit_segments` segments
/// per bus.  For bus `b_idx`, segment `seg_idx`, block `blk`:
/// `col = col_deficit_start + b_idx * max_deficit_segments * n_blks + seg_idx * n_blks + blk`
///
/// Buses with fewer than `max_deficit_segments` segments leave the trailing
/// slots at `[lower=0, upper=0, objective=0]` (from vec initialisation), which
/// the `HiGHS` presolver eliminates before the simplex phase.
fn fill_deficit_and_excess_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (b_idx, bus) in ctx.buses.iter().enumerate() {
        let bp = ctx.resolved.penalties.bus_penalties(b_idx, stage_idx);
        for (seg_idx, segment) in bus.deficit_segments.iter().enumerate() {
            for blk in 0..layout.n_blks {
                let col_def = layout.deficit_col(b_idx, seg_idx, blk);
                let block_hours = stage.blocks[blk].duration_hours;
                bufs.col_upper[col_def] = segment.depth_mw.unwrap_or(f64::INFINITY);
                bufs.objective[col_def] = segment.cost_per_mwh * block_hours;
            }
        }
        for blk in 0..layout.n_blks {
            let col_exc = layout
                .indexer
                .block_grid()
                .flat(layout.col_excess_start(), b_idx, blk);
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.col_upper[col_exc] = f64::INFINITY;
            bufs.objective[col_exc] = bp.excess_cost * block_hours;
        }
    }
}

/// Inflow non-negativity slack columns (`sigma_inf_h`), one per hydro.
///
/// Bounds `[0, +inf)` come from vec initialisation; only objective needs writing.
/// Per-plant cost from the penalty cascade.
fn fill_inflow_slack_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    total_stage_hours: f64,
    bufs: &mut ColumnBufs<'_>,
) {
    if ctx.has_penalty {
        for h_idx in 0..layout.n_h {
            let col = layout.col_inflow_slack_start() + h_idx;
            let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
            bufs.objective[col] = hp.inflow_nonnegativity_cost * total_stage_hours;
        }
    }
}

/// FPHA generation columns (`g_{h,k}`): one per FPHA hydro per block.
///
/// Bounds: `[0, max_generation_mw]`.  Objective: `0.0` (the global
/// `turbined_cost` is applied on the turbine column for every hydro).
fn fill_fpha_generation_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (local_idx, &h_idx) in layout.fpha_hydro_indices.iter().enumerate() {
        let hb = ctx.resolved.bounds.hydro_bounds(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.generation_col(local_idx, blk);
            bufs.col_lower[col] = 0.0;
            bufs.col_upper[col] = hb.max_generation_mw;
        }
    }
}

/// Evaporation columns: 3 per evaporation hydro (evaporation outflow,
/// `f_evap_plus`, `f_evap_minus`).
///
/// All three columns are stage-level (not per-block).  The evaporation-outflow
/// column is bounded symmetrically `[-q_max, +q_max]` so a negative value can
/// absorb net rainfall input on the lake surface; `f_evap_plus` and
/// `f_evap_minus` are bounded `[0, +inf)`.  The evaporation-outflow column
/// carries zero objective cost (evaporation flow itself is not penalised).
/// `f_evap_plus` and `f_evap_minus` carry
/// `evaporation_violation_cost * total_stage_hours` so that the solver is
/// penalised for violating the linearised evaporation constraint.
fn fill_evaporation_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    total_stage_hours: f64,
    bufs: &mut ColumnBufs<'_>,
) {
    for (local_idx, &h_idx) in layout.evap_hydro_indices.iter().enumerate() {
        let col_evaporation_flow = layout.evap_flow_col(local_idx);
        let col_f_plus = layout.evap_f_plus_col(local_idx);
        let col_f_minus = layout.evap_f_minus_col(local_idx);
        // Signed flow: a negative evaporation outflow reads as net rainfall input (inflow).
        // Bound: [-q_max, +q_max] where
        // q_max = |intercept_m3s + volume_slope_m3s_per_hm3 * v_max| * margin.
        match ctx.evaporation_models.model(h_idx) {
            EvaporationModel::Linearized { coefficients, .. } => {
                let coeff = &coefficients[stage_idx];
                let hb = ctx.resolved.bounds.hydro_bounds(h_idx, stage_idx);
                let q_max_abs = (coeff.intercept_m3s
                    + coeff.volume_slope_m3s_per_hm3 * hb.max_storage_hm3)
                    .abs()
                    * EVAPORATION_FLOW_SAFETY_MARGIN;
                bufs.col_lower[col_evaporation_flow] = -q_max_abs;
                bufs.col_upper[col_evaporation_flow] = q_max_abs;
            }
            EvaporationModel::None => {
                // Should never happen: evap_hydro_indices only contains linearized hydros.
                debug_assert!(
                    false,
                    "evap_hydro_indices contains hydro {h_idx} but model is None"
                );
                continue;
            }
        }
        bufs.col_lower[col_f_plus] = 0.0;
        bufs.col_upper[col_f_plus] = f64::INFINITY;
        bufs.col_lower[col_f_minus] = 0.0;
        bufs.col_upper[col_f_minus] = f64::INFINITY;
        // Violation cost: read directional costs from resolved penalties.
        // Evaporation outflow (offset 0) keeps objective = 0.0 (already the vec initialisation default).
        // f_evap_plus = under-evaporation (evaporated less than target).
        // f_evap_minus = over-evaporation (evaporated more than target).
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        bufs.objective[col_f_plus] = hp.evaporation_violation_neg_cost * total_stage_hours;
        bufs.objective[col_f_minus] = hp.evaporation_violation_pos_cost * total_stage_hours;
    }
}

/// Withdrawal violation slack columns — neg (under-withdrawal) and pos (over-withdrawal).
///
/// One stage-level column per hydro for each direction. In the water-balance row
/// the neg slack enters with coefficient `-zeta` and the pos slack with `+zeta`,
/// so the *realized* withdrawal removed from the reservoir is
/// `R = T - neg + pos`, where `T = water_withdrawal_m3s` (a signed schedule:
/// `T > 0` is a removal, `T < 0` an inter-basin return/addition).
///
/// ## Sign-aware under-delivery cap
///
/// Realized withdrawal must stay on its signed segment and must **not flip sign**
/// (a scheduled removal cannot become an injection, nor a scheduled addition a
/// removal): `R ∈ [0, T]` for `T > 0`, `R ∈ [T, 0]` for `T < 0`. The *under-delivery*
/// direction is the one that drags `R` toward — and potentially across — zero, so it
/// is capped at the target magnitude `|T|`; the *over-delivery* direction is left
/// unbounded (it pushes `R` further along its own sign, never across zero, and is the
/// solver's latitude to shed excess water through the withdrawal point):
///
/// - `T > 0`: under-delivery is `neg` (`R = T - neg`), capped `neg ≤ |T|` (floors `R ≥ 0`);
///   over-delivery is `pos`, left `+∞`.
/// - `T < 0`: under-delivery is `pos` (`R = T + pos`), capped `pos ≤ |T|` (floors `R ≤ 0`);
///   over-delivery is `neg`, left `+∞`.
/// - `T = 0`: both pinned to `0` so the columns are presolve-eliminated, preserving
///   identical behaviour to the pre-withdrawal implementation when withdrawal is absent.
fn fill_withdrawal_slack_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    total_stage_hours: f64,
    bufs: &mut ColumnBufs<'_>,
) {
    // Neg slacks: under-delivery for T>0 (cap at |T|), over-application for T<0 (unbounded).
    for h_idx in 0..layout.n_h {
        let col = layout.col_withdrawal_neg_start() + h_idx;
        let hb = ctx.resolved.bounds.hydro_bounds(h_idx, stage_idx);
        let t = hb.water_withdrawal_m3s;
        bufs.col_upper[col] = if t > 0.0 {
            t // under-delivery: floor R ≥ 0 by capping neg ≤ |T|
        } else if t < 0.0 {
            f64::INFINITY // over-application latitude (R further negative)
        } else {
            0.0 // no scheduled withdrawal: presolve-eliminate
        };
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        bufs.objective[col] = hp.water_withdrawal_violation_neg_cost * total_stage_hours;
    }
    // Pos slacks: over-delivery for T>0 (unbounded), under-delivery for T<0 (cap at |T|).
    for h_idx in 0..layout.n_h {
        let col = layout.col_withdrawal_pos_start() + h_idx;
        let hb = ctx.resolved.bounds.hydro_bounds(h_idx, stage_idx);
        let t = hb.water_withdrawal_m3s;
        bufs.col_upper[col] = if t > 0.0 {
            f64::INFINITY // over-withdrawal latitude (R further positive)
        } else if t < 0.0 {
            -t // under-delivery: floor R ≤ 0 by capping pos ≤ |T|
        } else {
            0.0 // no scheduled withdrawal: presolve-eliminate
        };
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        bufs.objective[col] = hp.water_withdrawal_violation_pos_cost * total_stage_hours;
    }
}

/// One operational-violation slack family, addressing a disjoint `n_h * n_blks`
/// column range. The variant selects, via `match`, all three axes that distinguish
/// the families: the resolved-bound activation predicate, the `StageLayout` column
/// accessor, and the `HydroStagePenalties` cost field. Every predicate reads the
/// **resolved per-stage** bound (`ctx.resolved.bounds.hydro_bounds(h_idx, stage_idx)`), never
/// the entity declaration on `ctx.hydros[h_idx]` — the declaration ignores per-stage
/// overrides and would silently mis-activate columns while still compiling.
#[derive(Clone, Copy)]
enum BlockSlackFamily {
    /// `sigma_outflow_below_{h,k}`: active (unbounded above) iff the resolved
    /// `min_outflow_m3s > 0.0`; pinned to `[0, 0]` otherwise.
    OutflowBelow,
    /// `sigma_outflow_above_{h,k}`: active (unbounded above) iff the resolved
    /// `max_outflow_m3s` is `Some` (an `Option::is_some()` check, NOT `> 0.0` — a
    /// `Some(0.0)` cap still activates the column); pinned to `[0, 0]` otherwise.
    OutflowAbove,
    /// `sigma_turbine_below_{h,k}`: active (unbounded above) iff the resolved
    /// `min_turbined_m3s > 0.0`; pinned to `[0, 0]` otherwise.
    TurbineBelow,
    /// `sigma_generation_below_{h,k}`: active (unbounded above) iff the resolved
    /// `min_generation_mw > 0.0`; pinned to `[0, 0]` otherwise.
    GenerationBelow,
}

/// Operational violation slack columns: 4 families of `n_h * n_blks` columns.
///
/// Drives [`fill_block_family`] once per family; each family writes to a disjoint
/// column range, so the call order does not affect the result.
fn fill_operational_slack_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    fill_block_family(
        ctx,
        stage,
        stage_idx,
        layout,
        bufs,
        BlockSlackFamily::OutflowBelow,
    );
    fill_block_family(
        ctx,
        stage,
        stage_idx,
        layout,
        bufs,
        BlockSlackFamily::OutflowAbove,
    );
    fill_block_family(
        ctx,
        stage,
        stage_idx,
        layout,
        bufs,
        BlockSlackFamily::TurbineBelow,
    );
    fill_block_family(
        ctx,
        stage,
        stage_idx,
        layout,
        bufs,
        BlockSlackFamily::GenerationBelow,
    );
}

/// Fill one operational-violation slack family's `n_h * n_blks` columns.
///
/// For each hydro the activation is decided once from the resolved per-stage bound;
/// for each block the column index comes from the family's `StageLayout` accessor and
/// the objective coefficient from the family's `HydroStagePenalties` cost field scaled
/// by the block duration. `col_lower[col]` is left at the vec default `0.0`.
fn fill_block_family(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
    family: BlockSlackFamily,
) {
    for h_idx in 0..layout.n_h {
        let hb = ctx.resolved.bounds.hydro_bounds(h_idx, stage_idx);
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        let active = match family {
            BlockSlackFamily::OutflowBelow => hb.min_outflow_m3s > 0.0,
            BlockSlackFamily::OutflowAbove => hb.max_outflow_m3s.is_some(),
            BlockSlackFamily::TurbineBelow => hb.min_turbined_m3s > 0.0,
            BlockSlackFamily::GenerationBelow => hb.min_generation_mw > 0.0,
        };
        let cost = match family {
            BlockSlackFamily::OutflowBelow => hp.outflow_violation_below_cost,
            BlockSlackFamily::OutflowAbove => hp.outflow_violation_above_cost,
            BlockSlackFamily::TurbineBelow => hp.turbined_violation_below_cost,
            BlockSlackFamily::GenerationBelow => hp.generation_violation_below_cost,
        };
        for blk in 0..layout.n_blks {
            let col = match family {
                BlockSlackFamily::OutflowBelow => layout.outflow_below_col(h_idx, blk),
                BlockSlackFamily::OutflowAbove => layout.outflow_above_col(h_idx, blk),
                BlockSlackFamily::TurbineBelow => layout.turbine_below_col(h_idx, blk),
                BlockSlackFamily::GenerationBelow => layout.generation_below_col(h_idx, blk),
            };
            bufs.col_upper[col] = if active { f64::INFINITY } else { 0.0 };
            bufs.objective[col] = cost * stage.blocks[blk].duration_hours;
        }
    }
}

/// NCS generation columns: one per active NCS per block.
///
/// `col_lower[col] = if ncs.allow_curtailment { 0 } else { col_upper[col] }`.
/// `col_upper[col] = available_gen * ncs_factor`.
/// `objective[col] = -curtailment_cost * block_hours` (negative incentivises generation).
///
/// These template values govern only when NCS noise is **non-stochastic**
/// (`n_stochastic_ncs == 0`). When stochastic NCS is active, both bounds
/// are rebuilt per scenario by `transform_ncs_noise` to scale with the
/// realized availability ratio `α = clamp(mean + std·η, 0, 1)`; the
/// template values are overwritten via `set_col_bounds` before each stage
/// solve. With `allow_curtailment == true` (the default) the column is
/// fully curtailable (bit-identical to the pre-flag behaviour); with
/// `allow_curtailment == false` the column is pinned to the available
/// level on every stage (must-run: non-simulated aggregate generation
/// pre-netted from load).
fn fill_ncs_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (ncs_local, &ncs_sys_idx) in layout.active_ncs_indices.iter().enumerate() {
        let ncs = &ctx.non_controllable_sources[ncs_sys_idx];
        let avail_gen = ctx
            .resolved
            .resolved_ncs_bounds
            .available_generation(ncs_sys_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout
                .indexer
                .block_grid()
                .flat(layout.col_ncs_start, ncs_local, blk);
            let factor = ctx
                .resolved
                .resolved_ncs_factors
                .factor(ncs_sys_idx, stage_idx, blk);
            let upper = avail_gen * factor;
            bufs.col_upper[col] = upper;
            bufs.col_lower[col] = if ncs.allow_curtailment { 0.0 } else { upper };
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col] = -ncs.curtailment_cost * block_hours;
        }
    }
}

/// Pumping-flow columns per station per block.
///
/// One column per ID-sorted pumping station per block, block-major
/// (`col_pumping_start + p_idx * n_blks + blk`). Bounds are the resolved
/// `[min_flow_m3s, max_flow_m3s]` for the `(station, stage)` pair; objective
/// cost is zero (pumping carries no direct cost — its electrical cost enters
/// through the bus load balance in the power-coupling pass, not here).
///
/// Iterates `ctx.pumping_stations` in slot order — the canonical ID-sorted slice
/// from `System::pumping_stations` — so the local index `p_idx` matches both the
/// column-block position and the `ctx.resolved.bounds.pumping_bounds(p_idx, …)` slot. This
/// upholds the declaration-order bit-determinism rule without depending on
/// declaration order.
pub(super) fn fill_pumping_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for p_idx in 0..ctx.pumping_stations.len() {
        let pb = ctx.resolved.bounds.pumping_bounds(p_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout
                .indexer
                .block_grid()
                .flat(layout.col_pumping_start, p_idx, blk);
            bufs.col_lower[col] = pb.min_flow_m3s;
            bufs.col_upper[col] = pb.max_flow_m3s;
            // objective[col] = 0.0 — already zero from vec initialisation.
        }
    }
}

/// Z-inflow columns: free variables for realized total inflow per hydro.
///
/// `col_lower = -inf`, `col_upper = +inf`, `objective = 0.0` (no direct cost).
fn fill_z_inflow_columns(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    for h_idx in 0..layout.n_h {
        let col = layout.col_z_inflow_start() + h_idx;
        bufs.col_lower[col] = f64::NEG_INFINITY;
        bufs.col_upper[col] = f64::INFINITY;
        // objective[col] = 0.0 — already zero from vec initialisation.
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod diversion_bound_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::hydro::{DiversionChannel, HydroGenerationModel};
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, CascadeTopology, ContractStageBounds, EntityId, Hydro,
        HydroStageBounds, HydroStagePenalties, LineStageBounds, PenaltiesCountsSpec,
        PenaltiesDefaults, PumpingStageBounds, ResolvedBounds, ResolvedExchangeFactors,
        ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties,
        ThermalStageBounds,
    };
    use cobre_core::{BusStagePenalties, LineStagePenalties, NcsStagePenalties};
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, ProductionModelSet, ResolvedProductionModel,
    };
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::{two_block_stage, zero_hydro_penalties};
    use super::{ColumnBufs, StageLayout, TemplateBuildCtx, fill_diversion_columns};

    // Declaration-time diversion capacity on the entity. The test makes the
    // resolved per-stage override distinct from this so the two reads disagree:
    // reading the entity directly (the pre-fix bug) would yield this value.
    const DECLARATION_MAX_FLOW_M3S: f64 = 200.0;
    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;

    /// One run-of-river hydro carrying a declaration-time diversion channel with
    /// `DECLARATION_MAX_FLOW_M3S` capacity and a `ConstantProductivity` generation
    /// model (so the layout reserves no FPHA/evaporation columns).
    fn diverting_hydro() -> Hydro {
        Hydro {
            id: EntityId(1),
            name: "H1".to_string(),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: Some(DiversionChannel {
                downstream_id: EntityId(2),
                max_flow_m3s: DECLARATION_MAX_FLOW_M3S,
            }),
            filling: None,
            penalties: zero_hydro_penalties(),
        }
    }

    /// Build a `ResolvedBounds` table for one hydro across `N_STAGES` stages. The
    /// fixture seeds `max_diversion_m3s` to `None`; both test bodies overwrite that
    /// cell before asserting, so the fixture default never affects the result.
    fn bounds_one_hydro() -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 200.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 100.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 250.0,
                    max_diversion_m3s: None,
                    filling_inflow_m3s: 0.0,
                    water_withdrawal_m3s: 0.0,
                },
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                    cost_per_mwh: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        )
    }

    fn penalties_one_hydro() -> ResolvedPenalties {
        ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 1,
                n_buses: 0,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: HydroStagePenalties {
                    spillage_cost: 0.0,
                    diversion_cost: 0.0,
                    turbined_cost: 0.0,
                    storage_violation_below_cost: 0.0,
                    filling_target_violation_cost: 0.0,
                    turbined_violation_below_cost: 0.0,
                    outflow_violation_below_cost: 0.0,
                    outflow_violation_above_cost: 0.0,
                    generation_violation_below_cost: 0.0,
                    evaporation_violation_cost: 0.0,
                    water_withdrawal_violation_cost: 0.0,
                    water_withdrawal_violation_pos_cost: 0.0,
                    water_withdrawal_violation_neg_cost: 0.0,
                    evaporation_violation_pos_cost: 0.0,
                    evaporation_violation_neg_cost: 0.0,
                    inflow_nonnegativity_cost: 0.0,
                },
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        )
    }

    /// Owns the borrow targets for a one-hydro `TemplateBuildCtx`.
    struct DivFixtures {
        par_lp: PrecomputedPar,
        hydros: Vec<Hydro>,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: cobre_core::ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_exchange_factors: ResolvedExchangeFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
    }

    impl DivFixtures {
        fn new() -> Self {
            let hydros = vec![diverting_hydro()];
            let cascade = CascadeTopology::build(&hydros);
            Self {
                par_lp: PrecomputedPar::default(),
                hydros,
                cascade,
                bounds: bounds_one_hydro(),
                penalties: penalties_one_hydro(),
                production_models: ProductionModelSet::new(
                    vec![vec![
                        ResolvedProductionModel::ConstantProductivity {
                            productivity: 1.0
                        };
                        N_STAGES
                    ]],
                    1,
                    N_STAGES,
                ),
                evaporation_models: EvaporationModelSet::new(vec![EvaporationModel::None]),
                resolved_generic_bounds: cobre_core::ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_exchange_factors: ResolvedExchangeFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                },
            }
        }

        /// Set the per-stage resolved `max_diversion_m3s` override for hydro 0.
        fn set_resolved_diversion(&mut self, value: Option<f64>) {
            self.bounds.hydro_bounds_mut(0, STAGE_IDX).max_diversion_m3s = value;
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            let mut hydro_pos = BTreeMap::new();
            hydro_pos.insert(self.hydros[0].id, 0_usize);
            TemplateBuildCtx {
                hydros: &self.hydros,
                thermals: &[],
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_exchange_factors: &self.resolved_exchange_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos,
                thermal_pos: BTreeMap::new(),
                line_pos: BTreeMap::new(),
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: 0,
                diversion_upstream: HashMap::new(),
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                has_penalty: false,
                cumulative_discount_factors: vec![1.0],
                total_hours_per_stage: vec![744.0],
            }
        }
    }

    /// Run `fill_diversion_columns` against the fixture and return `col_upper`.
    fn run_fill(fixtures: &DivFixtures) -> (Vec<f64>, StageLayout) {
        let stage = two_block_stage(STAGE_IDX, [372.0, 372.0]);
        let ctx = fixtures.make_ctx();
        let layout = StageLayout::new(&ctx, &stage, STAGE_IDX);
        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_diversion_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);
        (col_upper, layout)
    }

    /// A per-stage resolved override distinct from the declaration value flows
    /// into every diversion block's `col_upper`. Fails against the pre-fix code,
    /// which read `hydro.diversion.max_flow_m3s` (= `DECLARATION_MAX_FLOW_M3S`).
    #[test]
    fn resolved_diversion_override_flows_into_col_upper() {
        let override_value = 37.5;
        assert!(
            (override_value - DECLARATION_MAX_FLOW_M3S).abs() > f64::EPSILON,
            "override must differ from the declaration value for the test to bite"
        );
        let mut fixtures = DivFixtures::new();
        fixtures.set_resolved_diversion(Some(override_value));

        let (col_upper, layout) = run_fill(&fixtures);
        for blk in 0..layout.n_blks {
            let col = layout.col_diversion_start() + blk;
            assert_eq!(
                col_upper[col], override_value,
                "blk {blk}: col_upper[{col}] must equal the resolved override {override_value}"
            );
        }
    }

    /// No per-stage override preserves the inert default: `col_upper` equals the
    /// declaration `hydro.diversion.max_flow_m3s`. At the resolved-table level
    /// "no override" is `Some(declaration)` — the resolver seeds the declaration
    /// value into the cell and only overwrites it when a per-stage row supplies a
    /// different `Some(v)`, so this models what the resolver actually produces.
    #[test]
    fn no_override_diversion_preserves_declaration_default() {
        let mut fixtures = DivFixtures::new();
        // Seed the resolved cell with the declaration default, exactly as the
        // resolver does for a diverting hydro with no per-stage override.
        fixtures.set_resolved_diversion(Some(DECLARATION_MAX_FLOW_M3S));

        let (col_upper, layout) = run_fill(&fixtures);
        for blk in 0..layout.n_blks {
            let col = layout.col_diversion_start() + blk;
            assert_eq!(
                col_upper[col], DECLARATION_MAX_FLOW_M3S,
                "blk {blk}: col_upper[{col}] must equal the declaration default \
                 {DECLARATION_MAX_FLOW_M3S} when no override tightens it"
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod block_family_slack_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, CascadeTopology, ContractStageBounds,
        EntityId, Hydro, HydroStageBounds, HydroStagePenalties, LineStageBounds,
        LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
        PumpingStageBounds, ResolvedBounds, ResolvedExchangeFactors, ResolvedLoadFactors,
        ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, ProductionModelSet, ResolvedProductionModel,
    };
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::{two_block_stage, zero_hydro_penalties};
    use super::{ColumnBufs, StageLayout, TemplateBuildCtx, fill_operational_slack_columns};

    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;
    const N_HYDROS: usize = 6;
    const BLOCK_HOURS: [f64; 2] = [300.0, 444.0];

    /// One hydro's intended predicate state and the four distinct objective costs
    /// it carries. The costs differ per family and per hydro so a cost-field
    /// cross-wiring in the driver's `match` is observable in the assertions.
    struct HydroSpec {
        min_outflow_m3s: f64,
        max_outflow_m3s: Option<f64>,
        min_turbined_m3s: f64,
        min_generation_mw: f64,
        outflow_below_cost: f64,
        outflow_above_cost: f64,
        turbined_below_cost: f64,
        generation_below_cost: f64,
    }

    /// Six hydros spanning every predicate state plus an all-inert row. Hydro 2
    /// carries `max_outflow_m3s = Some(0.0)` to lock the `is_some()` semantics: a
    /// `> 0.0` comparison would wrongly deactivate its outflow-above column.
    fn hydro_specs() -> [HydroSpec; N_HYDROS] {
        [
            // H0: outflow-below active only.
            HydroSpec {
                min_outflow_m3s: 12.0,
                max_outflow_m3s: None,
                min_turbined_m3s: 0.0,
                min_generation_mw: 0.0,
                outflow_below_cost: 1.0,
                outflow_above_cost: 2.0,
                turbined_below_cost: 3.0,
                generation_below_cost: 4.0,
            },
            // H1: outflow-above active via Some(positive).
            HydroSpec {
                min_outflow_m3s: 0.0,
                max_outflow_m3s: Some(50.0),
                min_turbined_m3s: 0.0,
                min_generation_mw: 0.0,
                outflow_below_cost: 5.0,
                outflow_above_cost: 6.0,
                turbined_below_cost: 7.0,
                generation_below_cost: 8.0,
            },
            // H2: outflow-above active via Some(0.0) — the is_some() lock.
            HydroSpec {
                min_outflow_m3s: 0.0,
                max_outflow_m3s: Some(0.0),
                min_turbined_m3s: 0.0,
                min_generation_mw: 0.0,
                outflow_below_cost: 9.0,
                outflow_above_cost: 10.0,
                turbined_below_cost: 11.0,
                generation_below_cost: 12.0,
            },
            // H3: turbine-below active only.
            HydroSpec {
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_turbined_m3s: 7.0,
                min_generation_mw: 0.0,
                outflow_below_cost: 13.0,
                outflow_above_cost: 14.0,
                turbined_below_cost: 15.0,
                generation_below_cost: 16.0,
            },
            // H4: generation-below active only.
            HydroSpec {
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_turbined_m3s: 0.0,
                min_generation_mw: 9.0,
                outflow_below_cost: 17.0,
                outflow_above_cost: 18.0,
                turbined_below_cost: 19.0,
                generation_below_cost: 20.0,
            },
            // H5: all four inert.
            HydroSpec {
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_turbined_m3s: 0.0,
                min_generation_mw: 0.0,
                outflow_below_cost: 21.0,
                outflow_above_cost: 22.0,
                turbined_below_cost: 23.0,
                generation_below_cost: 24.0,
            },
        ]
    }

    /// Independent constant-productivity hydro (no FPHA/evaporation columns).
    fn fixture_hydro(id: i32) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_hydro_penalties(),
        }
    }

    fn zero_hydro_stage_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            max_diversion_m3s: None,
            filling_inflow_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn zero_hydro_stage_penalties() -> HydroStagePenalties {
        HydroStagePenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 0.0,
        }
    }

    /// Build the resolved bound and penalty tables, writing each hydro's predicate
    /// state and four costs into its resolved cell (the driver reads the resolved
    /// table, so the per-stage cells — not the entity declarations — carry the state).
    fn resolved_tables(specs: &[HydroSpec; N_HYDROS]) -> (ResolvedBounds, ResolvedPenalties) {
        let mut bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: N_HYDROS,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: zero_hydro_stage_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
                    cost_per_mwh: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 0.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let mut penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: N_HYDROS,
                n_buses: 0,
                n_lines: 0,
                n_ncs: 0,
                n_stages: N_STAGES,
            },
            &PenaltiesDefaults {
                hydro: zero_hydro_stage_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );
        for (h_idx, spec) in specs.iter().enumerate() {
            let hb = bounds.hydro_bounds_mut(h_idx, STAGE_IDX);
            hb.min_outflow_m3s = spec.min_outflow_m3s;
            hb.max_outflow_m3s = spec.max_outflow_m3s;
            hb.min_turbined_m3s = spec.min_turbined_m3s;
            hb.min_generation_mw = spec.min_generation_mw;
            let hp = penalties.hydro_penalties_mut(h_idx, STAGE_IDX);
            hp.outflow_violation_below_cost = spec.outflow_below_cost;
            hp.outflow_violation_above_cost = spec.outflow_above_cost;
            hp.turbined_violation_below_cost = spec.turbined_below_cost;
            hp.generation_violation_below_cost = spec.generation_below_cost;
        }
        (bounds, penalties)
    }

    /// Owns the borrow targets for a multi-hydro `TemplateBuildCtx`.
    struct SlackFixtures {
        par_lp: PrecomputedPar,
        hydros: Vec<Hydro>,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: cobre_core::ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_exchange_factors: ResolvedExchangeFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
    }

    impl SlackFixtures {
        fn new(specs: &[HydroSpec; N_HYDROS]) -> Self {
            let hydros: Vec<Hydro> = (0..N_HYDROS).map(|i| fixture_hydro(i as i32 + 1)).collect();
            let cascade = CascadeTopology::build(&hydros);
            let (bounds, penalties) = resolved_tables(specs);
            Self {
                par_lp: PrecomputedPar::default(),
                hydros,
                cascade,
                bounds,
                penalties,
                production_models: ProductionModelSet::new(
                    vec![
                        vec![
                            ResolvedProductionModel::ConstantProductivity { productivity: 1.0 };
                            N_STAGES
                        ];
                        N_HYDROS
                    ],
                    N_HYDROS,
                    N_STAGES,
                ),
                evaporation_models: EvaporationModelSet::new(vec![
                    EvaporationModel::None;
                    N_HYDROS
                ]),
                resolved_generic_bounds: cobre_core::ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_exchange_factors: ResolvedExchangeFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                },
            }
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            let mut hydro_pos = BTreeMap::new();
            for (i, h) in self.hydros.iter().enumerate() {
                hydro_pos.insert(h.id, i);
            }
            TemplateBuildCtx {
                hydros: &self.hydros,
                thermals: &[],
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                resolved: ResolvedTables {
                    bounds: &self.bounds,
                    penalties: &self.penalties,
                    resolved_generic_bounds: &self.resolved_generic_bounds,
                    resolved_load_factors: &self.resolved_load_factors,
                    resolved_exchange_factors: &self.resolved_exchange_factors,
                    resolved_ncs_bounds: &self.resolved_ncs_bounds,
                    resolved_ncs_factors: &self.resolved_ncs_factors,
                    resolved_parameters: &self.resolved_parameters,
                },
                hydro_pos,
                thermal_pos: BTreeMap::new(),
                line_pos: BTreeMap::new(),
                bus_pos: BTreeMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: 0,
                diversion_upstream: HashMap::new(),
                n_hydros: N_HYDROS,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                has_penalty: false,
                cumulative_discount_factors: vec![1.0],
                total_hours_per_stage: vec![BLOCK_HOURS[0] + BLOCK_HOURS[1]],
            }
        }
    }

    /// One family's expected contract: its name, the activation predicate over a
    /// `HydroSpec`, the `StageLayout` column accessor, and the expected cost field.
    struct FamilyCheck {
        name: &'static str,
        predicate: fn(&HydroSpec) -> bool,
        accessor: fn(&StageLayout, usize, usize) -> usize,
        cost_of: fn(&HydroSpec) -> f64,
    }

    /// Building the operational-violation slack columns through the enum-dispatched
    /// driver reproduces the per-family contract for every (hydro, block): `col_upper`
    /// is `+∞` exactly when the family's resolved predicate holds (else `0.0`),
    /// `objective` is the family's resolved cost times the block duration, and
    /// `col_lower` stays at the `0.0` default. Hydro 2's `Some(0.0)` cap locks the
    /// `is_some()` outflow-above semantics against a `> 0.0` regression.
    #[test]
    fn block_family_driver_matches_legacy_slack_fills() {
        let specs = hydro_specs();
        let fixtures = SlackFixtures::new(&specs);
        let stage = two_block_stage(STAGE_IDX, [BLOCK_HOURS[0], BLOCK_HOURS[1]]);
        let ctx = fixtures.make_ctx();
        let layout = StageLayout::new(&ctx, &stage, STAGE_IDX);

        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_operational_slack_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);

        // Each family's predicate, accessor, and cost, spelled out explicitly.
        let families = [
            FamilyCheck {
                name: "outflow_below",
                predicate: |s| s.min_outflow_m3s > 0.0,
                accessor: StageLayout::outflow_below_col,
                cost_of: |s| s.outflow_below_cost,
            },
            FamilyCheck {
                name: "outflow_above",
                predicate: |s| s.max_outflow_m3s.is_some(),
                accessor: StageLayout::outflow_above_col,
                cost_of: |s| s.outflow_above_cost,
            },
            FamilyCheck {
                name: "turbine_below",
                predicate: |s| s.min_turbined_m3s > 0.0,
                accessor: StageLayout::turbine_below_col,
                cost_of: |s| s.turbined_below_cost,
            },
            FamilyCheck {
                name: "generation_below",
                predicate: |s| s.min_generation_mw > 0.0,
                accessor: StageLayout::generation_below_col,
                cost_of: |s| s.generation_below_cost,
            },
        ];

        // The block loop iterates BLOCK_HOURS directly; assert the layout agrees so a
        // fixture/layout block-count drift cannot silently skip blocks.
        assert_eq!(layout.n_blks, BLOCK_HOURS.len());
        for family in &families {
            let name = family.name;
            for (h_idx, spec) in specs.iter().enumerate() {
                let active = (family.predicate)(spec);
                let cost = (family.cost_of)(spec);
                for (blk, &hours) in BLOCK_HOURS.iter().enumerate() {
                    let col = (family.accessor)(&layout, h_idx, blk);
                    let expected_upper = if active { f64::INFINITY } else { 0.0 };
                    assert_eq!(
                        col_upper[col], expected_upper,
                        "{name} h{h_idx} blk{blk}: col_upper[{col}] expected {expected_upper}"
                    );
                    let expected_obj = cost * hours;
                    assert_eq!(
                        objective[col], expected_obj,
                        "{name} h{h_idx} blk{blk}: objective[{col}] expected {expected_obj}"
                    );
                    assert_eq!(
                        col_lower[col], 0.0,
                        "{name} h{h_idx} blk{blk}: col_lower[{col}] must stay at 0.0"
                    );
                }
            }
        }
    }
}
