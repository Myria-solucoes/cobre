use cobre_core::{CoefficientRef, ConstraintSense, Stage};

use crate::generic_constraints::resolve_variable_ref;
use crate::hydro_models::{EvaporationModel, FphaPlane, ResolvedProductionModel};

use super::layout::{StageLayout, TemplateBuildCtx};
use super::{EVAPORATION_FLOW_SAFETY_MARGIN, M3S_TO_HM3};

/// Mutable column-bound and objective buffers shared by all fill helpers.
///
/// Passed by mutable reference to each `fill_*_columns` helper so that the
/// orchestrator `fill_stage_columns` call sites stay on a single line each.
/// This is an analogue of `LpMatrixBuffers` for the column-fill path.
struct ColumnBufs<'a> {
    col_lower: &'a mut [f64],
    col_upper: &'a mut [f64],
    objective: &'a mut [f64],
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
    zero_anticipated_delivery_thermal_cost(ctx, stage_idx, layout, b);
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
fn fill_anticipated_state_out_columns(
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
            let col = layout.col_thermal_start() + t_idx * layout.n_blks + blk;
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
        // (already the vec initialization default).
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

/// Zero out per-block thermal objective coefficients for anticipated thermals
/// at their delivery stages.
///
/// At every stage where the fishing equality is active for anticipated plant `i`,
/// the thermal's per-block generation cost is zeroed. This prevents double-counting:
/// the generation is already priced at the decision stage via
/// `fill_anticipated_decision_objective`; the delivery-stage LP must consume it at
/// zero marginal cost.
///
/// Must be called AFTER `fill_thermal_columns`, which writes the standard
/// non-zero cost for all thermals at all stages. This function overwrites that
/// cost for anticipated thermals at their delivery stages only.
///
/// The assignment `= 0.0` (not `*= 0.0`) is deliberate: a clean write that
/// survives floating-point anomalies. Dividing zero by `COST_SCALE_FACTOR`
/// (the post-loop in `build_single_stage_template`) still gives zero.
fn zero_anticipated_delivery_thermal_cost(
    ctx: &TemplateBuildCtx<'_>,
    _stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    let n_blks = layout.n_blks;
    for local_idx in 0..ctx.n_anticipated {
        let thermal_idx = ctx.anticipated_thermal_indices[local_idx];
        for blk in 0..n_blks {
            let col = layout.col_thermal_start() + thermal_idx * n_blks + blk;
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
            let col_exc = layout.col_excess_start() + b_idx * layout.n_blks + blk;
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
        let col = layout.col_withdrawal_neg_start + h_idx;
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
        let col = layout.col_withdrawal_pos_start + h_idx;
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
            let col = layout.col_ncs_start + ncs_local * layout.n_blks + blk;
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
fn fill_pumping_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for p_idx in 0..ctx.pumping_stations.len() {
        let pb = ctx.resolved.bounds.pumping_bounds(p_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.col_pumping_start + p_idx * layout.n_blks + blk;
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

/// Walk every FPHA hyperplane row of the stage, invoking `visit` once per
/// `(FPHA hydro, block, plane)` triple with the resolved plane and its LP row.
///
/// This is the single owner of the FPHA row-cursor arithmetic. Both the bounds
/// fill ([`fill_fpha_rows`]) and the coefficient fill ([`fill_fpha_entries`])
/// drive off this walker, so the cursor advance and the row formula live in one
/// place: a one-sided edit to either fill is no longer possible, which would
/// otherwise land the row bounds and the matrix coefficients on different rows.
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
fn for_each_fpha_plane<F>(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    mut visit: F,
) where
    F: FnMut(usize, usize, usize, usize, &FphaPlane, usize),
{
    let n_blks = layout.n_blks;
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
                let row = fpha_block_start + blk * n_planes + p_idx;
                visit(local_idx, h_idx, blk, p_idx, plane, row);
            }
        }
        fpha_block_start += n_blks * n_planes;
    }
}

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
    fill_operational_violation_rows(
        ctx,
        stage,
        stage_idx,
        layout,
        &mut row_lower,
        &mut row_upper,
    );
    fill_anticipated_fishing_rows(
        ctx,
        stage,
        stage_idx,
        layout,
        &mut row_lower,
        &mut row_upper,
    );
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
            let row = layout.row_load_balance_start() + b_idx * layout.n_blks + blk;
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
/// ([`fill_fpha_entries`]), so the static upper bound carries only the `intercept`.
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
        if let EvaporationModel::Linearized { coefficients, .. } =
            ctx.evaporation_models.model(h_idx)
        {
            debug_assert!(
                stage_idx < coefficients.len(),
                "stage index {stage_idx} out of bounds for evaporation coefficients (len = {})",
                coefficients.len()
            );
            let intercept_m3s = coefficients[stage_idx].intercept_m3s;
            let row = layout.row_evap_start + local_idx;
            row_lower[row] = intercept_m3s;
            row_upper[row] = intercept_m3s;
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
    _stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    row_lower: &mut [f64],
    row_upper: &mut [f64],
) {
    // All four operational-violation row families share the same per-hydro bound
    // lookup, so the bound is read once per hydro and the four families' bounds are
    // written from it. Each family targets `row_lower`/`row_upper` by its own
    // computed row index, so the visit order is irrelevant to the result.
    //   min-outflow   (>=): LHS + sigma >= min_outflow_m3s
    //   max-outflow   (<=): LHS - sigma <= max_outflow_m3s
    //   min-turbine   (>=): LHS + sigma >= min_turbined_m3s
    //   min-generation(>=): LHS + sigma >= min_generation_mw
    for h_idx in 0..layout.n_h {
        let hb = ctx.resolved.bounds.hydro_bounds(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let row = layout.row_min_outflow_start + h_idx * layout.n_blks + blk;
            row_lower[row] = hb.min_outflow_m3s;
            row_upper[row] = f64::INFINITY;
        }
        for blk in 0..layout.n_blks {
            let row = layout.row_max_outflow_start + h_idx * layout.n_blks + blk;
            row_lower[row] = f64::NEG_INFINITY;
            row_upper[row] = hb.max_outflow_m3s.unwrap_or(f64::INFINITY);
        }
        for blk in 0..layout.n_blks {
            let row = layout.row_min_turbine_start + h_idx * layout.n_blks + blk;
            row_lower[row] = hb.min_turbined_m3s;
            row_upper[row] = f64::INFINITY;
        }
        for blk in 0..layout.n_blks {
            let row = layout.row_min_generation_start + h_idx * layout.n_blks + blk;
            row_lower[row] = hb.min_generation_mw;
            row_upper[row] = f64::INFINITY;
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
    _stage: &Stage,
    _stage_idx: usize,
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
        if !layout
            .indexer
            .is_anticipated_decision_active(local_idx, stage_idx, n_stages)
        {
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

/// Fill CSC matrix entries for anticipated-fishing equality constraints.
///
/// For each anticipated plant `i` (always-active predicate), writes:
/// - `(row, +block_hours[blk])` on each per-block thermal column
///   `col_thermal_start + thermal_idx * n_blks + blk` (`LHS`, `MWh`).
/// - `(row, -block_hours_total)` on the anticipated-state slot-0 column
///   `col_anticipated_state_start + local_idx` (`RHS` coupling: `MW` × h = `MWh`).
///
/// The constraint enforces that the total generated energy (sum over blocks) equals
/// the committed power level (slot 0 content) scaled to `MWh`.
///
/// When `n_anticipated == 0`, this function is a no-op.
pub(super) fn fill_anticipated_fishing_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    _stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    for local_idx in 0..ctx.n_anticipated {
        let row = layout.anticipated.row_anticipated_fishing_start + local_idx;
        let thermal_idx = ctx.anticipated_thermal_indices[local_idx];
        let mut block_hours_total: f64 = 0.0;
        for blk in 0..n_blks {
            let col_gen = layout.col_thermal_start() + thermal_idx * n_blks + blk;
            let block_hours = stage.blocks[blk].duration_hours;
            col_entries[col_gen].push((row, block_hours));
            block_hours_total += block_hours;
        }
        let col_state = layout.col_anticipated_state_start() + local_idx;
        col_entries[col_state].push((row, -block_hours_total));
    }
    debug_assert_eq!(
        ctx.n_anticipated, layout.anticipated.n_anticipated_fishing_rows,
        "fill_anticipated_fishing_entries: row count must equal n_anticipated"
    );
}

/// Fill CSC entries for the `anticipated_state_out` definition equality rows.
///
/// For each active anticipated plant `i` (`stage_idx + K_i < n_stages`),
/// emits TWO CSC entries at the definition row:
/// - `(row, +1.0)` on `col_anticipated_state_out_start + i`
/// - `(row, -1.0)` on `col_anticipated_decision_start + i`
///
/// Encodes the equality `anticipated_state_out[i] − decision_col[i] = 0`.
/// Row bounds are filled by `fill_anticipated_state_out_def_rows`. The
/// final CSC ordering is enforced by the per-column
/// `sort_unstable_by_key(|&(row, _)| row)` pass in
/// `build_single_stage_template`, so the relative order of the two pushes here
/// does not matter for correctness.
///
/// Inactive plants emit no entries.
///
/// No-op when `n_anticipated == 0` or when no plant is active.
pub(super) fn fill_anticipated_state_out_def_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_stages = ctx.resolved.bounds.n_stages();
    let mut active_pos: usize = 0;
    for local_idx in 0..ctx.n_anticipated {
        if !layout
            .indexer
            .is_anticipated_decision_active(local_idx, stage_idx, n_stages)
        {
            continue;
        }
        let row = layout.anticipated.row_anticipated_state_out_def_start + active_pos;
        let col_state_out = layout.anticipated.col_anticipated_state_out_start + local_idx;
        let col_decision = layout.anticipated.col_anticipated_decision_start + local_idx;
        col_entries[col_state_out].push((row, 1.0));
        col_entries[col_decision].push((row, -1.0));
        active_pos += 1;
    }
    debug_assert_eq!(
        active_pos, layout.anticipated.n_anticipated_state_out_def_rows,
        "fill_anticipated_state_out_def_entries: active_pos mismatch at stage {stage_idx}"
    );
}

/// Fill water-balance row entries into `col_entries`.
///
/// Writes entries for the water-balance rows (outgoing/incoming
/// storage, turbine, spillage, upstream cascade, and AR lag
/// dynamics). State pinning has moved to column bounds (Phase 1); the
/// row-equality diagonals previously written here are gone.
pub(super) fn fill_state_and_water_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_h = layout.n_h;
    let n_blks = layout.n_blks;
    let lag_order = layout.lag_order;
    let zeta = layout.zeta;
    let row_water = layout.row_water_balance_start();
    let col_storage_in_start = layout.col_storage_in_start();
    let col_inflow_lags_start = layout.col_inflow_lags_start();

    // Water balance: outgoing storage (+1), incoming storage (-1),
    // turbine/spillage (+tau), upstream turbine/spillage (-tau),
    // and AR lag dynamics (-ζ*ψ).
    for h_idx in 0..n_h {
        let hydro = &ctx.hydros[h_idx];
        let row = row_water + h_idx;
        col_entries[h_idx].push((row, 1.0));
        col_entries[col_storage_in_start + h_idx].push((row, -1.0));
        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col_turbine = layout.turbine_col(h_idx, blk);
            col_entries[col_turbine].push((row, tau_h));
            let col_spillage = layout.spillage_col(h_idx, blk);
            col_entries[col_spillage].push((row, tau_h));
            // Diversion outflow: this hydro's diversion column enters its own
            // water balance with +tau (outflow), same sign as turbine/spillage.
            let col_diversion = layout.diversion_col(h_idx, blk);
            col_entries[col_diversion].push((row, tau_h));
            // Cascade inflow: upstream turbine/spillage enter with -tau.
            for &up_id in ctx.cascade.upstream(hydro.id) {
                if let Some(&u_idx) = ctx.hydro_pos.get(&up_id) {
                    col_entries[layout.turbine_col(u_idx, blk)].push((row, -tau_h));
                    col_entries[layout.spillage_col(u_idx, blk)].push((row, -tau_h));
                }
            }
            // Diversion inflow: for each hydro that diverts TO this hydro, its
            // diversion column enters this hydro's water balance with -tau.
            if let Some(sources) = ctx.diversion_upstream.get(&hydro.id) {
                for &d_idx in sources {
                    let col_div = layout.diversion_col(d_idx, blk);
                    col_entries[col_div].push((row, -tau_h));
                }
            }
        }
        if ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == n_h {
            let psi = ctx.par_lp.psi_slice(stage_idx, h_idx);
            for (lag, &psi_val) in psi.iter().enumerate() {
                if psi_val != 0.0 && lag < lag_order {
                    let col = col_inflow_lags_start + lag * n_h + h_idx;
                    col_entries[col].push((row, -zeta * psi_val));
                }
            }
        }
    }

    // Inflow non-negativity slack: sigma_inf_h enters water balance with -ζ.
    if ctx.has_penalty {
        for h_idx in 0..n_h {
            let col = layout.col_inflow_slack_start() + h_idx;
            let row = row_water + h_idx;
            col_entries[col].push((row, -zeta));
        }
    }

    // Evaporation: the per-hydro evaporation-outflow column enters water balance with +ζ.
    // Evaporation is an outflow (water leaving the reservoir), so its
    // coefficient matches the turbine/spillage sign convention (positive).
    for (local_idx, &h_idx) in layout.evap_hydro_indices.iter().enumerate() {
        let col_evaporation_flow = layout.evap_flow_col(local_idx);
        let row = row_water + h_idx;
        col_entries[col_evaporation_flow].push((row, zeta));
    }

    // Under-withdrawal slack (neg): adds water back to the balance.
    // When the reservoir cannot sustain the full scheduled withdrawal, the neg slack
    // absorbs the difference, reducing the effective withdrawal in that stage.
    for h_idx in 0..n_h {
        let col = layout.col_withdrawal_neg_start + h_idx;
        let row = row_water + h_idx;
        col_entries[col].push((row, -zeta));
    }

    // Over-withdrawal slack (pos): removes additional water from the balance.
    // When the solver withdraws more than the target, the pos slack accounts for
    // the excess withdrawal at a penalty cost.
    for h_idx in 0..n_h {
        let col = layout.col_withdrawal_pos_start + h_idx;
        let row = row_water + h_idx;
        col_entries[col].push((row, zeta));
    }
}

/// Fill pumping-flow water-balance row entries into `col_entries`.
///
/// A pumping station moves water from its SOURCE reservoir to its DESTINATION
/// reservoir. The pumped flow column therefore enters two water rows per block:
///
/// - the SOURCE hydro's water row with `+tau_h` (water LEAVES the source — the
///   same sign a turbine/spillage outflow carries in a plant's own row, see
///   [`fill_state_and_water_entries`]);
/// - the DESTINATION hydro's water row with `−tau_h` (water ARRIVES at the
///   destination — the same sign cascade-upstream inflow carries in the
///   downstream row, see [`fill_state_and_water_entries`]).
///
/// `tau_h` is the identical `stage.blocks[blk].duration_hours * M3S_TO_HM3`
/// expression used by turbine/spillage in [`fill_state_and_water_entries`] —
/// the same arithmetic on the same operands, so the coefficient is bit-identical
/// across the two sites; computing a differently-rounded τ would desynchronise
/// pumping from the cascade water terms.
///
/// Stations are iterated in `ctx.pumping_stations` slot order (the canonical
/// ID-sorted slice). A source or destination hydro id absent from `ctx.hydro_pos`
/// skips only that side's entry — the present side is still written, and no panic
/// occurs (semantic validation of the references is a separate concern).
pub(super) fn fill_pumping_water_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let row_water = layout.row_water_balance_start();
    for (p_idx, station) in ctx.pumping_stations.iter().enumerate() {
        // `validate_pumping_station_refs` (run from `SystemBuilder::build()`)
        // guarantees both refs resolve on a validated `System`, so on the
        // production path both `Option`s are `Some`. The per-side `if let
        // Some(...)` guards below are defense-in-depth for direct-construction
        // test paths that bypass validation — never a production branch. A
        // one-sided resolve writes a feasible-but-wrong half water coupling, so
        // the guards must not be promoted to an unconditional index/expect.
        let source = ctx.hydro_pos.get(&station.source_hydro_id).copied();
        let destination = ctx.hydro_pos.get(&station.destination_hydro_id).copied();
        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col = layout.col_pumping_start + p_idx * n_blks + blk;
            if let Some(s_idx) = source {
                col_entries[col].push((row_water + s_idx, tau_h));
            }
            if let Some(d_idx) = destination {
                col_entries[col].push((row_water + d_idx, -tau_h));
            }
        }
    }
}

/// Fill load-balance entries into `col_entries`.
///
/// Writes entries for hydro turbine generation, thermal generation,
/// line forward/reverse flows, pumping power consumption, and deficit/excess slacks.
///
/// For FPHA hydros the generation variable `g_{h,k}` (in `col_generation_start`)
/// enters the load balance with coefficient +1.0 instead of `rho * turbine_col`.
/// For constant-productivity hydros the original `rho * turbine_col` behavior is unchanged.
///
/// Pumping power `Eb = consumption_mw_per_m3s · Qb` is a negative injection on the
/// station's bus: the `pumping_flow` column enters the load-balance row with
/// `−consumption_mw_per_m3s` — the same negative sign a line carries into its
/// source bus (`col_fwd` → `−1.0` below), not the `+1.0` of generation. There is
/// no separate power column; the coefficient scales the SAME flow column, so a
/// positive coefficient (treating pumping as generation) would credit the bus for
/// power the station consumes.
pub(super) fn fill_load_balance_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let row_load = layout.row_load_balance_start();

    for (h_idx, hydro) in ctx.hydros.iter().enumerate() {
        if let Some(local_idx) = layout.fpha_local_index[h_idx] {
            debug_assert!(
                matches!(
                    ctx.production_models.model(h_idx, stage_idx),
                    ResolvedProductionModel::Fpha { .. }
                ),
                "FPHA local-index table inconsistent with production model for hydro {h_idx}"
            );
            if let Some(&b_idx) = ctx.bus_pos.get(&hydro.bus_id) {
                for blk in 0..n_blks {
                    let row = row_load + b_idx * n_blks + blk;
                    let col = layout.generation_col(local_idx, blk);
                    col_entries[col].push((row, 1.0));
                }
            }
        } else {
            let rho = match ctx.production_models.model(h_idx, stage_idx) {
                ResolvedProductionModel::ConstantProductivity { productivity } => *productivity,
                ResolvedProductionModel::Fpha { .. } => {
                    unreachable!(
                        "non-FPHA branch reached for FPHA resolved model at hydro {h_idx}"
                    );
                }
            };
            if let Some(&b_idx) = ctx.bus_pos.get(&hydro.bus_id) {
                for blk in 0..n_blks {
                    let row = row_load + b_idx * n_blks + blk;
                    let col = layout.turbine_col(h_idx, blk);
                    col_entries[col].push((row, rho));
                }
            }
        }
    }

    for (t_idx, thermal) in ctx.thermals.iter().enumerate() {
        if let Some(&b_idx) = ctx.bus_pos.get(&thermal.bus_id) {
            for blk in 0..n_blks {
                let row = row_load + b_idx * n_blks + blk;
                let col = layout.col_thermal_start() + t_idx * n_blks + blk;
                col_entries[col].push((row, 1.0));
            }
        }
    }

    for (l_idx, line) in ctx.lines.iter().enumerate() {
        let src_idx = ctx.bus_pos.get(&line.source_bus_id).copied();
        let tgt_idx = ctx.bus_pos.get(&line.target_bus_id).copied();
        for blk in 0..n_blks {
            let col_fwd = layout.line_fwd_col(l_idx, blk);
            let col_rev = layout.line_rev_col(l_idx, blk);
            if let Some(tgt) = tgt_idx {
                let row = row_load + tgt * n_blks + blk;
                col_entries[col_fwd].push((row, 1.0));
                col_entries[col_rev].push((row, -1.0));
            }
            if let Some(src) = src_idx {
                let row = row_load + src * n_blks + blk;
                col_entries[col_fwd].push((row, -1.0));
                col_entries[col_rev].push((row, 1.0));
            }
        }
    }

    // Pumping power: negative injection on the station's bus. Iterate the
    // canonical ID-sorted `pumping_stations` slice so `p_idx` matches the
    // column block; a bus id absent from `bus_pos` skips that station with no
    // entry (semantic validation is a separate concern).
    for (p_idx, station) in ctx.pumping_stations.iter().enumerate() {
        if let Some(&b_idx) = ctx.bus_pos.get(&station.bus_id) {
            for blk in 0..n_blks {
                let row = row_load + b_idx * n_blks + blk;
                let col = layout.col_pumping_start + p_idx * n_blks + blk;
                col_entries[col].push((row, -station.consumption_mw_per_m3s));
            }
        }
    }

    for (b_idx, bus) in ctx.buses.iter().enumerate() {
        for blk in 0..n_blks {
            let row = row_load + b_idx * n_blks + blk;
            for seg_idx in 0..bus.deficit_segments.len() {
                let col_def = layout.deficit_col(b_idx, seg_idx, blk);
                col_entries[col_def].push((row, 1.0));
            }
            let col_exc = layout.col_excess_start() + b_idx * n_blks + blk;
            col_entries[col_exc].push((row, -1.0));
        }
    }
}

/// Fill FPHA hyperplane constraint entries into `col_entries`.
///
/// For each FPHA hydro `h` at this stage, for each block `k`, for each
/// hyperplane `m`, adds matrix entries to FPHA row `r(h,k,m)`:
///
/// ```text
/// g_{h,k}  column:  +1.0              (generation variable)
/// v        column:  -gamma_v/2         (outgoing storage)
/// v_in     column:  -gamma_v/2         (incoming storage; fixed by storage-fixing row)
/// q_{h,k}  column:  -gamma_q           (turbined flow)
/// s_{h,k}  column:  -gamma_s           (spillage)
/// ```
///
/// These entries implement `g - gamma_v/2*v - gamma_v/2*v_in - gamma_q*q - gamma_s*s <= gamma_0`,
/// where `gamma_0` is already encoded in the row upper bound set by `fill_stage_rows`.
///
/// FPHA uses **average** storage `(V_in + V_out)/2`, so `-gamma_v/2` lands on
/// BOTH the outgoing-storage column (`v = h_idx`) and the incoming-storage column
/// (`v_in = col_storage_in_start + h_idx`). Putting `-gamma_v/2` on `v` (`V_out`)
/// alone compiles and passes single-plane single-hydro tests, but understates
/// generation by the `V_in` head term — the wrong-but-compiling alternative that
/// deterministic case D06 pins against.
///
/// Driven by [`for_each_fpha_plane`] so the matrix coefficients and the row
/// bounds set by [`fill_fpha_rows`] share one row cursor.
pub(super) fn fill_fpha_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let col_storage_in_start = layout.col_storage_in_start();
    for_each_fpha_plane(
        ctx,
        stage_idx,
        layout,
        |local_idx, h_idx, blk, _p_idx, plane, row| {
            let col_v = h_idx; // outgoing storage column
            let col_v_in = col_storage_in_start + h_idx; // incoming storage column
            let col_q = layout.turbine_col(h_idx, blk);
            let col_s = layout.spillage_col(h_idx, blk);
            let col_g = layout.generation_col(local_idx, blk);

            // g_{h,k} column: +1.0
            col_entries[col_g].push((row, 1.0));
            // v (outgoing storage): -gamma_v/2 — average-storage term, also on v_in below.
            col_entries[col_v].push((row, -plane.gamma_v / 2.0));
            // v_in (incoming storage, fixed by storage-fixing row): -gamma_v/2.
            col_entries[col_v_in].push((row, -plane.gamma_v / 2.0));
            // q_{h,k} (turbine): -gamma_q
            col_entries[col_q].push((row, -plane.gamma_q));
            // s_{h,k} (spillage): -gamma_s
            col_entries[col_s].push((row, -plane.gamma_s));
        },
    );
}

/// Fill CSC matrix entries for the evaporation constraint rows.
///
/// For each evaporation hydro `h` at local position `local_idx`, the equality row
/// `row_evap_start + local_idx` encodes:
///
/// ```text
/// evaporation_flow column:  +1.0
/// v_h     column:  -volume_slope_m3s_per_hm3 / 2   (outgoing storage)
/// v_in_h  column:  -volume_slope_m3s_per_hm3 / 2   (incoming storage; fixed by storage-fixing row)
/// f_plus  column:  +1.0
/// f_minus column:  -1.0
/// ```
///
/// These entries implement
/// `evaporation_flow - volume_slope_m3s_per_hm3/2*v - volume_slope_m3s_per_hm3/2*v_in + f_plus - f_minus = intercept_m3s`,
/// where `intercept_m3s` is already encoded in the row bounds set by
/// `fill_stage_rows`. When `v_in` is fixed to value `V`, the effective RHS
/// becomes `intercept_m3s + volume_slope_m3s_per_hm3/2 * V`, which matches the
/// linearized evaporation at the average volume `(v + V) / 2`.
pub(super) fn fill_evaporation_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let col_storage_in_start = layout.col_storage_in_start();

    for (local_idx, &h_idx) in layout.evap_hydro_indices.iter().enumerate() {
        let coeff = match ctx.evaporation_models.model(h_idx) {
            EvaporationModel::Linearized { coefficients, .. } => {
                debug_assert!(
                    stage_idx < coefficients.len(),
                    "evap_hydro_indices contains hydro {h_idx} but coefficients length {} \
                     is less than stage_idx {}",
                    coefficients.len(),
                    stage_idx
                );
                match coefficients.get(stage_idx) {
                    Some(c) => *c,
                    None => continue,
                }
            }
            EvaporationModel::None => {
                // Should never happen: evap_hydro_indices only contains linearized hydros.
                debug_assert!(
                    false,
                    "evap_hydro_indices contains hydro {h_idx} but model is None"
                );
                continue;
            }
        };

        let col_evaporation_flow = layout.evap_flow_col(local_idx);
        let col_f_plus = layout.evap_f_plus_col(local_idx);
        let col_f_minus = layout.evap_f_minus_col(local_idx);
        let col_v = h_idx;
        let col_v_in = col_storage_in_start + h_idx;

        let row = layout.row_evap_start + local_idx;

        col_entries[col_evaporation_flow].push((row, 1.0));
        col_entries[col_v].push((row, -coeff.volume_slope_m3s_per_hm3 / 2.0));
        col_entries[col_v_in].push((row, -coeff.volume_slope_m3s_per_hm3 / 2.0));
        col_entries[col_f_plus].push((row, 1.0));
        col_entries[col_f_minus].push((row, -1.0));
    }
}

/// Fill CSC matrix entries, row bounds, and slack column data for all active
/// generic constraint rows at this stage.
///
/// For each active `(constraint, block)` pair recorded in
/// `layout.generic_constraint_rows`:
///
/// 1. Sets `row_lower` / `row_upper` for the generic constraint row according
///    to the constraint sense:
///    - `<=`: `row_lower = -INF`, `row_upper = bound`
///    - `>=`: `row_lower = bound`, `row_upper = +INF`
///    - `==`: `row_lower = bound`, `row_upper = bound`
///
/// 2. Iterates over the constraint expression terms, calls
///    `resolve_variable_ref` for each `LinearTerm`, and pushes
///    `(row_index, coefficient * multiplier)` entries into `col_entries`.
///
/// 3. When `slack.enabled = true`, sets slack column bounds to `[0, +INF)` and
///    objective to `penalty * block_hours`:
///    - `<=`: one slack column `s_g` with CSC entry `(row, -1.0)`.
///    - `>=`: one slack column `s_g` with CSC entry `(row, +1.0)`.
///    - `==`: two slack columns `s_g_plus` and `s_g_minus` with CSC entries
///      `(row, +1.0)` and `(row, -1.0)` respectively.
///
/// Unknown entity IDs in variable refs produce zero contributions (the empty
/// vec returned by `resolve_variable_ref` is skipped), which is the
/// defense-in-depth fallback for referential validation gaps.
/// Mutable LP matrix buffers for stage template construction.
///
/// Groups the column and row arrays that are filled during template building.
pub(super) struct LpMatrixBuffers<'a> {
    /// CSC column entries (column index -> list of (row, coefficient)).
    pub(super) col_entries: &'a mut [Vec<(usize, f64)>],
    /// Column upper bounds.
    pub(super) col_upper: &'a mut [f64],
    /// Objective function coefficients.
    pub(super) objective: &'a mut [f64],
    /// Row lower bounds.
    pub(super) row_lower: &'a mut [f64],
    /// Row upper bounds.
    pub(super) row_upper: &'a mut [f64],
}

pub(super) fn fill_generic_constraint_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    buffers: &mut LpMatrixBuffers<'_>,
) {
    let col_entries = &mut *buffers.col_entries;
    let col_upper = &mut *buffers.col_upper;
    let objective = &mut *buffers.objective;
    let row_lower = &mut *buffers.row_lower;
    let row_upper = &mut *buffers.row_upper;
    if layout.n_generic_rows == 0 {
        return;
    }

    // Use the indexer already built and cached on the layout; avoids rebuilding
    // and cloning the anticipated metadata vecs on every template build call.
    let positions = crate::generic_constraints::EntityPositionMaps {
        hydro: &ctx.hydro_pos,
        thermal: &ctx.thermal_pos,
        bus: &ctx.bus_pos,
        line: &ctx.line_pos,
    };
    // Cascade topology + diversion-into map for the HydroInflow total-inflow
    // arm; both already borrowed on the build context.
    let cascade_refs = crate::generic_constraints::CascadeRefs {
        cascade: ctx.cascade,
        diversion_upstream: &ctx.diversion_upstream,
    };
    // Pumping column start + station data for the PumpingFlow/PumpingPower arms.
    // `col_pumping_start` is the real reserved range on the `StageLayout` being
    // built — NOT `StageIndexer::pumping_flow` (a permanent `0..0` sentinel).
    let pumping_refs = crate::generic_constraints::PumpingRefs {
        col_pumping_start: layout.col_pumping_start,
        pumping_stations: ctx.pumping_stations,
        pumping_pos: &ctx.pumping_pos,
    };

    for (entry_idx, entry) in layout.generic_constraint_rows.iter().enumerate() {
        let row = layout.row_generic_start + entry_idx;
        let constraint = &ctx.generic_constraints[entry.constraint_idx];
        // A collapsed stage-level row is priced by the stage's total hours
        // (it stands in for one row per block); a per-block row by its own
        // block's hours. The total equals `penalty * Σ block_hours * D` either
        // way, so the collapse is penalty-conserving.
        let block_hours = if entry.is_stage_level {
            stage.blocks.iter().map(|b| b.duration_hours).sum()
        } else {
            stage.blocks[entry.block_idx].duration_hours
        };

        // 1. Set row bounds from sense and RHS bound value.
        match entry.sense {
            ConstraintSense::LessEqual => {
                row_lower[row] = f64::NEG_INFINITY;
                row_upper[row] = entry.bound;
            }
            ConstraintSense::GreaterEqual => {
                row_lower[row] = entry.bound;
                row_upper[row] = f64::INFINITY;
            }
            ConstraintSense::Equal => {
                row_lower[row] = entry.bound;
                row_upper[row] = entry.bound;
            }
        }

        // 2. Fill CSC matrix entries for each expression term.
        for term in &constraint.expression.terms {
            let pairs = resolve_variable_ref(
                &term.variable,
                entry.block_idx,
                layout.n_blks,
                stage_idx,
                &layout.indexer,
                ctx.production_models,
                &positions,
                &cascade_refs,
                &pumping_refs,
            );
            for (col, multiplier) in pairs {
                let coef = match term.coefficient {
                    CoefficientRef::Literal(v) => v,
                    CoefficientRef::Parameter(param_id) => {
                        ctx.resolved.resolved_parameters.get(param_id, stage_idx)
                    }
                };
                col_entries[col].push((row, coef * term.scale * multiplier));
            }
        }

        // 3. Set slack column bounds and CSC entries when slack is enabled.
        if let Some(plus_col) = entry.slack_plus_col {
            let penalty = constraint.slack.penalty.unwrap_or(0.0);
            let obj_coeff = penalty * block_hours;

            // plus slack: [0, +INF), penalised in objective.
            // col_lower is already 0.0 from vec initialisation.
            col_upper[plus_col] = f64::INFINITY;
            objective[plus_col] = obj_coeff;

            // CSC entry for plus slack depends on sense.
            match entry.sense {
                ConstraintSense::LessEqual => {
                    // LHS - s_g <= bound  →  slack enters with -1.0
                    col_entries[plus_col].push((row, -1.0));
                }
                ConstraintSense::GreaterEqual => {
                    // LHS + s_g >= bound  →  slack enters with +1.0
                    col_entries[plus_col].push((row, 1.0));
                }
                ConstraintSense::Equal => {
                    // LHS + s_g_plus - s_g_minus == bound  →  plus slack with +1.0
                    col_entries[plus_col].push((row, 1.0));
                }
            }

            // minus slack: only for equality constraints.
            if let Some(minus_col) = entry.slack_minus_col {
                // col_lower is already 0.0 from vec initialisation.
                col_upper[minus_col] = f64::INFINITY;
                objective[minus_col] = obj_coeff;
                // LHS + s_g_plus - s_g_minus == bound  →  minus slack with -1.0
                col_entries[minus_col].push((row, -1.0));
            }
        }
    }
}

/// Fill NCS generation entries into the load balance constraint rows.
///
/// For each active NCS `r` at block `k`, injects `+1.0` at the load balance
/// row of the connected bus, identical to thermal generation injection.
pub(super) fn fill_ncs_load_balance_entries(
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    for (ncs_local, &ncs_sys_idx) in layout.active_ncs_indices.iter().enumerate() {
        let ncs = &ctx.non_controllable_sources[ncs_sys_idx];
        let Some(&bus_idx) = ctx.bus_pos.get(&ncs.bus_id) else {
            // Unknown bus — should not happen with valid data, but defensive skip.
            continue;
        };
        for blk in 0..layout.n_blks {
            let col = layout.col_ncs_start + ncs_local * layout.n_blks + blk;
            let row = layout.row_load_balance_start() + bus_idx * layout.n_blks + blk;
            col_entries[col].push((row, 1.0));
        }
    }
}

/// Fill z-inflow definition constraint entries into `col_entries`.
///
/// For each hydro h, the z-inflow constraint is:
///   `z_h - sum_l[psi_l * lag_in[h,l]] = base_h + sigma_h * eta_h`
///
/// Matrix entries:
/// - Column `z_h`: coefficient `+1.0` in row `row_z_inflow_start + h`
/// - For each lag l with nonzero `psi_l`: column `inflow_lags.start + lag * n_h + h`
///   gets coefficient `-psi_l` in row `row_z_inflow_start + h`
///
/// Note: the lag column layout matches the LP builder convention (lag-major):
/// the column at `inflow_lags.start + lag * n_h + h` stores lag `l` of hydro `h`.
pub(super) fn fill_z_inflow_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_h = layout.n_h;
    let lag_order = layout.lag_order;
    let col_inflow_lags_start = layout.col_inflow_lags_start();

    for h_idx in 0..n_h {
        let row = layout.row_z_inflow_start() + h_idx;

        // z_h column: coefficient +1.0
        let col_z = layout.col_z_inflow_start() + h_idx;
        col_entries[col_z].push((row, 1.0));

        // Lag columns: coefficient -psi_l for each nonzero psi.
        // Uses lag-major layout (lag * n_h + h) matching the water-balance
        // AR dynamics entries in fill_state_and_water_entries.
        if ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == n_h {
            let psi = ctx.par_lp.psi_slice(stage_idx, h_idx);
            for (lag, &psi_val) in psi.iter().enumerate() {
                if psi_val != 0.0 && lag < lag_order {
                    let col = col_inflow_lags_start + lag * n_h + h_idx;
                    col_entries[col].push((row, -psi_val));
                }
            }
        }
    }
}

/// Fill CSC matrix entries for the 4 operational violation constraint families.
///
/// Each constraint links decision variables (turbine, spillage, diversion, generation)
/// to their respective slack columns via the constraint rows allocated in
/// [`StageLayout`].
///
/// - **Min outflow** (`>=`): `sum_blk[tau * (q + s + d)] + sigma_below >= RHS`
/// - **Max outflow** (`<=`): `sum_blk[tau * (q + s + d)] - sigma_above <= RHS`
/// - **Min turbine** (`>=`): `sum_blk[tau * q] + sigma_below >= RHS`
/// - **Min generation** (`>=`): `sum_blk[coeff * var * hours] + sigma_below >= RHS`
///   where `coeff * var` is `rho * q` for constant-productivity hydros or `g` for FPHA.
pub(super) fn fill_operational_violation_entries(
    ctx: &TemplateBuildCtx<'_>,
    _stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;

    for (h_idx, fpha_local_entry) in layout.fpha_local_index.iter().enumerate() {
        // ── Min outflow (per block): q + s + d + sigma >= min_outflow_m3s ───
        for blk in 0..n_blks {
            let row = layout.row_min_outflow_start + h_idx * n_blks + blk;
            let col_q = layout.turbine_col(h_idx, blk);
            col_entries[col_q].push((row, 1.0));
            let col_s = layout.spillage_col(h_idx, blk);
            col_entries[col_s].push((row, 1.0));
            let col_d = layout.diversion_col(h_idx, blk);
            col_entries[col_d].push((row, 1.0));
            let col_slack = layout.outflow_below_col(h_idx, blk);
            col_entries[col_slack].push((row, 1.0));
        }

        // ── Max outflow (per block): q + s + d - sigma <= max_outflow_m3s ───
        for blk in 0..n_blks {
            let row = layout.row_max_outflow_start + h_idx * n_blks + blk;
            let col_q = layout.turbine_col(h_idx, blk);
            col_entries[col_q].push((row, 1.0));
            let col_s = layout.spillage_col(h_idx, blk);
            col_entries[col_s].push((row, 1.0));
            let col_d = layout.diversion_col(h_idx, blk);
            col_entries[col_d].push((row, 1.0));
            let col_slack = layout.outflow_above_col(h_idx, blk);
            col_entries[col_slack].push((row, -1.0));
        }

        // ── Min turbine flow (per block): q + sigma >= min_turbined_m3s ─────
        for blk in 0..n_blks {
            let row = layout.row_min_turbine_start + h_idx * n_blks + blk;
            let col_q = layout.turbine_col(h_idx, blk);
            col_entries[col_q].push((row, 1.0));
            let col_slack = layout.turbine_below_col(h_idx, blk);
            col_entries[col_slack].push((row, 1.0));
        }

        // ── Min generation (per block): g + sigma >= min_generation_mw ──────
        if let Some(&local_fpha_idx) = fpha_local_entry.as_ref() {
            // FPHA: generation variable g_{h,blk} (already in MW).
            for blk in 0..n_blks {
                let row = layout.row_min_generation_start + h_idx * n_blks + blk;
                let col_g = layout.generation_col(local_fpha_idx, blk);
                col_entries[col_g].push((row, 1.0));
                let col_slack = layout.generation_below_col(h_idx, blk);
                col_entries[col_slack].push((row, 1.0));
            }
        } else {
            // Constant productivity: gen_k = rho * q_k (MW).
            // Always read rho from the resolved per-stage production model.
            let rho = match ctx.production_models.model(h_idx, stage_idx) {
                ResolvedProductionModel::ConstantProductivity { productivity } => *productivity,
                ResolvedProductionModel::Fpha { .. } => {
                    unreachable!(
                        "Fpha resolved model in ConstantProductivity LP path for hydro \
                         {h_idx}; validate production model assignment upstream"
                    );
                }
            };
            for blk in 0..n_blks {
                let row = layout.row_min_generation_start + h_idx * n_blks + blk;
                let col_q = layout.turbine_col(h_idx, blk);
                col_entries[col_q].push((row, rho));
                let col_slack = layout.generation_below_col(h_idx, blk);
                col_entries[col_slack].push((row, 1.0));
            }
        }
    }
}

/// Build the unsorted CSC matrix entries for one stage.
///
/// Returns one `Vec<(row, value)>` per column. Entries are in insertion
/// order; the caller is responsible for sorting by row index before
/// assembling the final CSC arrays (see `build_single_stage_template`).
pub(super) fn build_stage_matrix_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
) -> Vec<Vec<(usize, f64)>> {
    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];

    fill_state_and_water_entries(ctx, stage, stage_idx, layout, &mut col_entries);
    fill_pumping_water_entries(ctx, stage, layout, &mut col_entries);
    fill_anticipated_state_out_def_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_load_balance_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_ncs_load_balance_entries(ctx, layout, &mut col_entries);
    fill_fpha_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_evaporation_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_z_inflow_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_operational_violation_entries(ctx, stage, stage_idx, layout, &mut col_entries);
    fill_anticipated_fishing_entries(ctx, stage, stage_idx, layout, &mut col_entries);

    col_entries
}

/// Assemble CSC arrays from per-column entry lists.
///
/// Returns `(col_starts, row_indices, values)` in the format required by
/// `SolverInterface::load_model`.
pub(super) fn assemble_csc(col_entries: &[Vec<(usize, f64)>]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    // Contract: within each column the (row, _) entries are sorted by row index
    // (non-decreasing). The caller (`build_single_stage_template` CSC assembly)
    // owns the `sort_unstable_by_key(|&(row, _)| row)`; `assemble_csc` emits each
    // column's rows in iteration order and does NOT sort. Passing unsorted entries
    // produces CSC `row_indices` out of order within a column, which HiGHS/CLP may
    // silently misfactorize — the assert surfaces a missing caller-side sort rather
    // than masking it with an internal re-sort. debug-only: the release/hot path
    // pays no scan.
    debug_assert!(
        col_entries
            .iter()
            .all(|c| c.windows(2).all(|w| w[0].0 <= w[1].0)),
        "assemble_csc: each column's entries must be row-sorted (caller-owned sort)"
    );
    let num_cols = col_entries.len();
    let total_nz: usize = col_entries.iter().map(Vec::len).sum();
    let mut col_starts = Vec::with_capacity(num_cols + 1);
    let mut row_indices = Vec::with_capacity(total_nz);
    let mut values = Vec::with_capacity(total_nz);

    let mut offset: i32 = 0;
    for entries in col_entries {
        col_starts.push(offset);
        for &(row, val) in entries {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            row_indices.push(row as i32);
            values.push(val);
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        {
            offset += entries.len() as i32;
        }
    }
    col_starts.push(offset);

    (col_starts, row_indices, values)
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod assemble_csc_tests {
    use super::assemble_csc;

    /// Sorted columns assemble into the expected CSC triple without panicking.
    /// Columns: c0 = [(0, 1.0), (2, 2.0)], c1 = [], c2 = [(1, 3.0)].
    #[test]
    fn test_assemble_csc_sorted_columns_assemble_expected_csc() {
        let col_entries: Vec<Vec<(usize, f64)>> =
            vec![vec![(0, 1.0), (2, 2.0)], vec![], vec![(1, 3.0)]];

        let (col_starts, row_indices, values) = assemble_csc(&col_entries);

        // col_starts has num_cols + 1 entries; the prefix sum of per-column nnz.
        assert_eq!(col_starts, vec![0, 2, 2, 3]);
        assert_eq!(row_indices, vec![0, 2, 1]);
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    /// A column whose `(row, _)` pairs are out of order trips the `debug_assert`,
    /// proving the caller-owned-sort precondition is guarded in debug/test builds.
    #[test]
    #[should_panic(expected = "each column's entries must be row-sorted")]
    fn test_assemble_csc_unsorted_column_panics() {
        // Column 0 has rows 2 then 1 — strictly decreasing, violating the contract.
        let col_entries: Vec<Vec<(usize, f64)>> = vec![vec![(2, 1.0), (1, 2.0)]];

        let _ = assemble_csc(&col_entries);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests: parameter resolution in the LP builder
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(
    clippy::too_many_lines,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
mod parameter_resolution_tests {
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties, ConstraintExpression,
        ConstraintSense, ContractStageBounds, DeficitSegment, EntityId, GenericConstraint,
        HydroStageBounds, HydroStagePenalties, LineStageBounds, LineStagePenalties,
        NcsStagePenalties, ParameterKind, PenaltiesCountsSpec, PenaltiesDefaults,
        PumpingStageBounds, ResolvedBounds, ResolvedGenericConstraintBounds, ResolvedPenalties,
        ScalarParameter, SlackConfig, SystemBuilder, ThermalStageBounds,
    };
    use cobre_core::{LinearTerm, VariableRef};
    use cobre_stochastic::normal::precompute::PrecomputedNormal;
    use cobre_stochastic::par::precompute::PrecomputedPar;
    use std::collections::HashMap;

    use crate::energy_conversion::{EnergyConversionSet, build_hydro_energy_productivity_override};
    use crate::hydro_models::PrepareHydroModelsResult;
    use crate::inflow_method::InflowNonNegativityMethod;
    use crate::resolved_parameters::build_resolved_parameters;

    /// Return all CSC values stored at `(col, row)` in the template.
    fn csc_entries_at(t: &cobre_solver::StageTemplate, col: usize, row: usize) -> Vec<f64> {
        let start = t.col_starts[col] as usize;
        let end = t.col_starts[col + 1] as usize;
        t.row_indices[start..end]
            .iter()
            .zip(t.values[start..end].iter())
            .filter_map(|(&r, &v)| if r as usize == row { Some(v) } else { None })
            .collect()
    }

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
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
        }
    }

    fn default_hydro_penalties() -> HydroStagePenalties {
        HydroStagePenalties {
            spillage_cost: 0.01,
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
            inflow_nonnegativity_cost: 1000.0,
        }
    }

    /// Build a one-bus, one-thermal system with `n_stages` stages and one
    /// generic constraint. Each stage has a single block of 744 hours.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn one_thermal_n_stages(
        n_stages: usize,
        thermal_entity_id: EntityId,
        constraints: Vec<GenericConstraint>,
        bounds: ResolvedGenericConstraintBounds,
    ) -> cobre_core::System {
        use chrono::NaiveDate;
        use cobre_core::entities::thermal::Thermal;
        use cobre_core::scenario::LoadModel;
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let thermal = Thermal {
            id: thermal_entity_id,
            name: "T1".to_string(),
            bus_id: EntityId(1),
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            cost_per_mwh: 50.0,
            anticipated_config: None,
            entry_stage_id: None,
            exit_stage_id: None,
        };

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: Some(0),
                blocks: vec![Block {
                    index: 0,
                    name: "BLK0".to_string(),
                    duration_hours: 744.0,
                }],
                block_mode: BlockMode::Parallel,
                state_config: StageStateConfig {
                    storage: false,
                    inflow_lags: false,
                },
                risk_config: StageRiskConfig::Expectation,
                scenario_config: ScenarioSourceConfig {
                    branching_factor: 1,
                    noise_method: NoiseMethod::Saa,
                },
            })
            .collect();

        let load_models: Vec<_> = (0..n_stages)
            .map(|i| LoadModel {
                bus_id: EntityId(1),
                stage_id: i as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        let resolved_bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
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
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(vec![thermal])
            .stages(stages)
            .load_models(load_models)
            .bounds(resolved_bounds)
            .penalties(penalties)
            .generic_constraints(constraints)
            .resolved_generic_bounds(bounds)
            .build()
            .expect("one_thermal_n_stages: valid system")
    }

    /// Build templates for the given system using the supplied `ResolvedParameters`.
    fn make_templates(
        system: &cobre_core::System,
        resolved_params: &crate::resolved_parameters::ResolvedParameters,
    ) -> Vec<cobre_solver::StageTemplate> {
        let production = PrepareHydroModelsResult::default_from_system(system).production;
        let evaporation = PrepareHydroModelsResult::default_from_system(system).evaporation;
        crate::lp_builder::build_stage_templates(
            system,
            InflowNonNegativityMethod::None,
            &PrecomputedPar::default(),
            &PrecomputedNormal::default(),
            &production,
            &evaporation,
            resolved_params,
        )
        .expect("make_templates: valid")
        .templates
    }

    /// Build an empty `ResolvedParameters` table (no parameters) with a given
    /// stage-to-season mapping.
    fn empty_resolved_params(n_stages: usize) -> crate::resolved_parameters::ResolvedParameters {
        let stage_to_season: Vec<i32> = vec![0; n_stages];
        let ec = EnergyConversionSet::new(vec![], vec![], 0, n_stages);
        let override_table =
            build_hydro_energy_productivity_override(&[]).expect("empty override table");
        build_resolved_parameters(&[], &ec, &override_table, &[], &stage_to_season, n_stages)
            .expect("empty_resolved_params: valid")
    }

    /// Build a `ResolvedParameters` table containing a single `Constant` parameter.
    fn constant_param_resolved(
        param_id: EntityId,
        value: f64,
        n_stages: usize,
    ) -> crate::resolved_parameters::ResolvedParameters {
        let stage_to_season: Vec<i32> = vec![0; n_stages];
        let ec = EnergyConversionSet::new(vec![], vec![], 0, n_stages);
        let override_table =
            build_hydro_energy_productivity_override(&[]).expect("empty override table");
        let params = vec![ScalarParameter {
            id: param_id,
            name: format!("p{}", param_id.0),
            kind: ParameterKind::Constant { value },
        }];
        build_resolved_parameters(
            &params,
            &ec,
            &override_table,
            &[],
            &stage_to_season,
            n_stages,
        )
        .expect("constant_param_resolved: valid")
    }

    /// Build a `ResolvedParameters` table containing a single `PerStage` parameter.
    fn per_stage_param_resolved(
        param_id: EntityId,
        values: Vec<f64>,
    ) -> crate::resolved_parameters::ResolvedParameters {
        let n_stages = values.len();
        let stage_to_season: Vec<i32> = vec![0; n_stages];
        let ec = EnergyConversionSet::new(vec![], vec![], 0, n_stages);
        let override_table =
            build_hydro_energy_productivity_override(&[]).expect("empty override table");
        let params = vec![ScalarParameter {
            id: param_id,
            name: format!("p{}", param_id.0),
            kind: ParameterKind::PerStage { values },
        }];
        build_resolved_parameters(
            &params,
            &ec,
            &override_table,
            &[],
            &stage_to_season,
            n_stages,
        )
        .expect("per_stage_param_resolved: valid")
    }

    /// Make a generic constraint with a single `Parameter` term over a thermal.
    fn parameter_constraint(
        constraint_id: EntityId,
        param_id: EntityId,
        scale: f64,
        thermal_id: EntityId,
    ) -> GenericConstraint {
        GenericConstraint {
            id: constraint_id,
            name: format!("gc_{}", constraint_id.0),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::parameter(
                    param_id,
                    scale,
                    VariableRef::ThermalGeneration {
                        thermal_id,
                        block_id: None,
                    },
                )],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        }
    }

    /// Make a generic constraint with a single `Literal` term over a thermal.
    fn literal_constraint(
        constraint_id: EntityId,
        coef: f64,
        scale: f64,
        thermal_id: EntityId,
    ) -> GenericConstraint {
        GenericConstraint {
            id: constraint_id,
            name: format!("gc_lit_{}", constraint_id.0),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm {
                    coefficient: cobre_core::CoefficientRef::Literal(coef),
                    scale,
                    variable: VariableRef::ThermalGeneration {
                        thermal_id,
                        block_id: None,
                    },
                }],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        }
    }

    /// Build a `ResolvedGenericConstraintBounds` for a single constraint active
    /// at all `n_stages` stages (bound value `50.0`).
    fn bounds_for_n_stages(
        constraint_id: EntityId,
        n_stages: usize,
    ) -> ResolvedGenericConstraintBounds {
        let id_map: HashMap<i32, usize> = [(constraint_id.0, 0)].into_iter().collect();
        let rows: Vec<(i32, i32, Option<i32>, f64)> = (0..n_stages)
            .map(|s| (constraint_id.0, s as i32, None, 50.0_f64))
            .collect();
        ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter())
    }

    // ── Layout constants for 1-bus, 1-thermal, 1-block, 0-hydro ──────────────
    //
    // theta=col 0, decision_start=col 1
    // thermal col 0 block 0 → col 1
    // load_balance row 0 → generic constraint row at row 1
    const THERMAL_COL: usize = 1;
    const GENERIC_ROW: usize = 1;

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1: single stage, constant parameter, coefficient = param * scale * 1
    // ─────────────────────────────────────────────────────────────────────────

    /// A `CoefficientRef::Parameter` term is resolved against `ResolvedParameters`
    /// at LP-build time.
    ///
    /// Fixture: EntityId(7) → 3.0 (constant), scale 2.0.
    /// Expected CSC value: 3.0 * 2.0 * 1.0 = 6.0.
    #[test]
    fn parameter_coefficient_is_resolved_at_lp_build() {
        let param_id = EntityId(7);
        let thermal_id = EntityId(2);
        let constraint_id = EntityId(10);
        let scale = 2.0_f64;
        let param_value = 3.0_f64;

        let constraint = parameter_constraint(constraint_id, param_id, scale, thermal_id);
        let generic_bounds = bounds_for_n_stages(constraint_id, 1);

        let system = one_thermal_n_stages(1, thermal_id, vec![constraint], generic_bounds);
        let resolved = constant_param_resolved(param_id, param_value, 1);
        let templates = make_templates(&system, &resolved);

        let t = &templates[0];
        let entries = csc_entries_at(t, THERMAL_COL, GENERIC_ROW);
        assert!(
            !entries.is_empty(),
            "no CSC entry at (col={THERMAL_COL}, row={GENERIC_ROW}) for parameter term"
        );
        let total: f64 = entries.iter().sum();
        let expected = param_value * scale; // multiplier=1.0
        assert_eq!(
            total.to_bits(),
            expected.to_bits(),
            "expected {expected} (bit-exact), got {total}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2: three stages, per-stage parameter, each stage gets its own value
    // ─────────────────────────────────────────────────────────────────────────

    /// When the parameter has per-stage values, each stage template receives
    /// the stage-specific resolved coefficient.
    ///
    /// Fixture: EntityId(7) → [3.0, 7.0, 11.0], scale 2.0.
    /// Expected CSC values: stage 0 → 6.0, stage 1 → 14.0, stage 2 → 22.0.
    #[test]
    fn parameter_coefficient_per_stage_changes_lp() {
        let param_id = EntityId(7);
        let thermal_id = EntityId(2);
        let constraint_id = EntityId(10);
        let scale = 2.0_f64;
        let per_stage_values = vec![3.0_f64, 7.0, 11.0];
        let n_stages = per_stage_values.len();

        let constraint = parameter_constraint(constraint_id, param_id, scale, thermal_id);
        let generic_bounds = bounds_for_n_stages(constraint_id, n_stages);

        let system = one_thermal_n_stages(n_stages, thermal_id, vec![constraint], generic_bounds);
        let resolved = per_stage_param_resolved(param_id, per_stage_values.clone());
        let templates = make_templates(&system, &resolved);

        for (stage_idx, &param_val) in per_stage_values.iter().enumerate() {
            let t = &templates[stage_idx];
            let entries = csc_entries_at(t, THERMAL_COL, GENERIC_ROW);
            assert!(
                !entries.is_empty(),
                "no CSC entry at (col={THERMAL_COL}, row={GENERIC_ROW}) for stage {stage_idx}"
            );
            let total: f64 = entries.iter().sum();
            let expected = param_val * scale; // multiplier=1.0
            assert_eq!(
                total.to_bits(),
                expected.to_bits(),
                "stage {stage_idx}: expected {expected} (bit-exact), got {total}"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3: regression — Literal coefficients still work after wiring
    // ─────────────────────────────────────────────────────────────────────────

    /// `CoefficientRef::Literal(5.0)` with scale 3.0 must produce a CSC entry
    /// of exactly `5.0 * 3.0 * 1.0 = 15.0`, unchanged by the parameter-resolution
    /// wiring.
    #[test]
    fn literal_coefficient_still_works_after_wiring() {
        let thermal_id = EntityId(2);
        let constraint_id = EntityId(10);
        let coef = 5.0_f64;
        let scale = 3.0_f64;

        let constraint = literal_constraint(constraint_id, coef, scale, thermal_id);
        let generic_bounds = bounds_for_n_stages(constraint_id, 1);

        let system = one_thermal_n_stages(1, thermal_id, vec![constraint], generic_bounds);
        let resolved = empty_resolved_params(1);
        let templates = make_templates(&system, &resolved);

        let t = &templates[0];
        let entries = csc_entries_at(t, THERMAL_COL, GENERIC_ROW);
        assert!(
            !entries.is_empty(),
            "no CSC entry at (col={THERMAL_COL}, row={GENERIC_ROW}) for literal term"
        );
        let total: f64 = entries.iter().sum();
        let expected = coef * scale; // multiplier=1.0
        assert_eq!(
            total.to_bits(),
            expected.to_bits(),
            "expected {expected} (bit-exact), got {total}"
        );
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod zero_cost_tests {
    use std::collections::HashMap;

    use chrono::NaiveDate;
    use cobre_core::{
        Block, BlockMode, BoundsCountsSpec, BoundsDefaults, CascadeTopology, ContractStageBounds,
        HydroStageBounds, LineStageBounds, NoiseMethod, PumpingStageBounds, ResolvedBounds,
        ResolvedExchangeFactors, ResolvedGenericConstraintBounds, ResolvedLoadFactors,
        ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties, ScenarioSourceConfig, Stage,
        StageRiskConfig, StageStateConfig, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::{
        ColumnBufs, StageLayout, TemplateBuildCtx, build_stage_matrix_entries,
        fill_anticipated_fishing_entries, fill_anticipated_fishing_rows,
        fill_anticipated_state_out_columns, fill_anticipated_state_out_def_entries,
        fill_anticipated_state_out_def_rows, zero_anticipated_delivery_thermal_cost,
    };

    /// Build a minimal two-block `Stage` at the given index.
    fn two_block_stage(index: usize) -> Stage {
        Stage {
            index,
            id: index as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![
                Block {
                    index: 0,
                    name: "BLK0".to_string(),
                    duration_hours: 372.0,
                },
                Block {
                    index: 1,
                    name: "BLK1".to_string(),
                    duration_hours: 372.0,
                },
            ],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// Owns data for a context with anticipated thermals and zero other entities.
    struct AntFixtures {
        par_lp: PrecomputedPar,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_exchange_factors: ResolvedExchangeFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
    }

    impl AntFixtures {
        /// Build a minimal `ResolvedBounds` with zero entities and the given `n_stages`.
        ///
        /// `n_stages` must exceed the queried `stage_idx` (and each plant's
        /// delivery stage `stage_idx + K_i`) so the anticipated layout the
        /// fixture builds places every plant inside the study horizon
        /// `[0, n_stages)`.
        fn bounds_with_n_stages(n_stages: usize) -> ResolvedBounds {
            ResolvedBounds::new(
                &BoundsCountsSpec {
                    n_hydros: 0,
                    n_thermals: 0,
                    n_lines: 0,
                    n_pumping: 0,
                    n_contracts: 0,
                    n_stages,
                    k_max: 0,
                },
                &BoundsDefaults {
                    hydro: HydroStageBounds {
                        min_storage_hm3: 0.0,
                        max_storage_hm3: 0.0,
                        min_turbined_m3s: 0.0,
                        max_turbined_m3s: 0.0,
                        min_outflow_m3s: 0.0,
                        max_outflow_m3s: None,
                        min_generation_mw: 0.0,
                        max_generation_mw: 0.0,
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

        fn new() -> Self {
            Self {
                par_lp: PrecomputedPar::default(),
                cascade: CascadeTopology::build(&[]),
                bounds: ResolvedBounds::empty(),
                penalties: ResolvedPenalties::empty(),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_exchange_factors: ResolvedExchangeFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                },
                production_models: ProductionModelSet::new(vec![], 0, 1),
                evaporation_models: EvaporationModelSet::new(vec![]),
            }
        }

        fn make_ctx(
            &self,
            n_anticipated: usize,
            k_max: usize,
            anticipated_lead_stages: Vec<usize>,
            anticipated_thermal_indices: Vec<usize>,
            n_thermals: usize,
        ) -> TemplateBuildCtx<'_> {
            TemplateBuildCtx {
                hydros: &[],
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
                hydro_pos: HashMap::new(),
                thermal_pos: HashMap::new(),
                line_pos: HashMap::new(),
                bus_pos: HashMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: HashMap::new(),
                n_pumping: 0,
                diversion_upstream: HashMap::new(),
                n_hydros: 0,
                n_thermals,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated,
                k_max,
                anticipated_lead_stages,
                anticipated_thermal_indices,
                has_penalty: false,
                cumulative_discount_factors: vec![1.0],
                total_hours_per_stage: vec![744.0],
            }
        }
    }

    /// Zeroes cost for every anticipated plant (the fishing constraint is
    /// always active, one plant per iteration).
    /// At stage 2 with `K_i=[1, 5]` and `n_anticipated=2`: both plants zeroed.
    #[test]
    fn zero_anticipated_delivery_thermal_cost_zeroes_all_plants() {
        let mut fixtures = AntFixtures::new();
        // Override bounds so every plant's delivery stage `stage_idx + K_i`
        // falls inside the horizon `[0, n_stages)` when the layout is built.
        fixtures.bounds = AntFixtures::bounds_with_n_stages(10);
        let ctx = fixtures.make_ctx(
            2,          // n_anticipated
            5,          // k_max (max of [1, 5])
            vec![1, 5], // anticipated_lead_stages: K_0 = 1, K_1 = 5
            vec![0, 1], // anticipated_thermal_indices: positions in ctx.thermals
            2,          // n_thermals
        );
        let stage = two_block_stage(2);
        let layout = StageLayout::new(&ctx, &stage, 2);

        // Allocate objective buffer at full column width, pre-filled with a
        // sentinel so any un-zeroed entry is visible in the assertions.
        let mut objective = vec![42.0_f64; layout.num_cols];
        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };

        zero_anticipated_delivery_thermal_cost(&ctx, 2, &layout, &mut bufs);

        let n_blks = layout.n_blks;
        // Plant 0 (K_0=1): zeroed under always-active predicate.
        let thermal_idx_0 = ctx.anticipated_thermal_indices[0];
        for blk in 0..n_blks {
            let col = layout.col_thermal_start() + thermal_idx_0 * n_blks + blk;
            assert_eq!(
                bufs.objective[col], 0.0,
                "plant 0 must be zeroed at col {col}",
            );
        }
        // Plant 1 (K_1=5): also zeroed under always-active predicate.
        let thermal_idx_1 = ctx.anticipated_thermal_indices[1];
        for blk in 0..n_blks {
            let col = layout.col_thermal_start() + thermal_idx_1 * n_blks + blk;
            assert_eq!(
                bufs.objective[col], 0.0,
                "plant 1 must be zeroed at col {col}",
            );
        }
    }

    /// Fills rows for all anticipated plants (always-active predicate).
    /// At stage 2 with `K_i=[1, 5]` and `n_anticipated=2`: both plants are active,
    /// so exactly two fishing rows are written.
    #[test]
    fn fishing_rows_fill_all_plants() {
        let mut fixtures = AntFixtures::new();
        fixtures.bounds = AntFixtures::bounds_with_n_stages(10);
        let ctx = fixtures.make_ctx(2, 5, vec![1, 5], vec![0, 1], 2);
        let stage = two_block_stage(2);
        let layout = StageLayout::new(&ctx, &stage, 2);

        // Always-active: both plants active at stage 2 → two fishing rows.
        assert_eq!(
            layout.anticipated.n_anticipated_fishing_rows, 2,
            "expected n_anticipated_fishing_rows == 2, got {}",
            layout.anticipated.n_anticipated_fishing_rows
        );

        let mut row_lower = vec![f64::NAN; layout.num_rows];
        let mut row_upper = vec![f64::NAN; layout.num_rows];

        fill_anticipated_fishing_rows(&ctx, &stage, 2, &layout, &mut row_lower, &mut row_upper);

        // Both plants write a row with (0.0, 0.0) bounds.
        for local_idx in 0..layout.anticipated.n_anticipated_fishing_rows {
            let row = layout.anticipated.row_anticipated_fishing_start + local_idx;
            assert_eq!(
                row_lower[row], 0.0,
                "row_lower[{row}] (local_idx={local_idx}) expected 0.0, got {}",
                row_lower[row]
            );
            assert_eq!(
                row_upper[row], 0.0,
                "row_upper[{row}] (local_idx={local_idx}) expected 0.0, got {}",
                row_upper[row]
            );
        }
    }

    /// Always-active at `stage_idx = 0`: with `K = [1, 5]` and `n_anticipated = 2`,
    /// both plants are active even before their lead time elapses.
    /// Asserts `layout.anticipated.n_anticipated_fishing_rows == 2`, that both rows
    /// are filled with `(0.0, 0.0)` bounds, and that the anticipated-state
    /// slot-0 column carries the `-block_hours_total` coupling for both plants.
    #[test]
    fn fishing_rows_always_active_stage_zero() {
        let mut fixtures = AntFixtures::new();
        fixtures.bounds = AntFixtures::bounds_with_n_stages(10);
        let ctx = fixtures.make_ctx(2, 5, vec![1, 5], vec![0, 1], 2);
        let stage = two_block_stage(0);
        let layout = StageLayout::new(&ctx, &stage, 0);

        // Always-active: both plants are active at stage 0 → two fishing rows.
        assert_eq!(
            layout.anticipated.n_anticipated_fishing_rows, 2,
            "expected n_anticipated_fishing_rows == 2 at stage 0, got {}",
            layout.anticipated.n_anticipated_fishing_rows
        );

        let mut row_lower = vec![f64::NAN; layout.num_rows];
        let mut row_upper = vec![f64::NAN; layout.num_rows];

        fill_anticipated_fishing_rows(&ctx, &stage, 0, &layout, &mut row_lower, &mut row_upper);

        // Both plants write equality rows with (0.0, 0.0) bounds.
        for local_idx in 0..layout.anticipated.n_anticipated_fishing_rows {
            let row = layout.anticipated.row_anticipated_fishing_start + local_idx;
            assert_eq!(
                row_lower[row], 0.0,
                "row_lower[{row}] (local_idx={local_idx}) expected 0.0, got {}",
                row_lower[row]
            );
            assert_eq!(
                row_upper[row], 0.0,
                "row_upper[{row}] (local_idx={local_idx}) expected 0.0, got {}",
                row_upper[row]
            );
        }

        // CSC coupling: anticipated_state slot-0 column carries (row, -block_hours_total)
        // for each plant under the always-active predicate.
        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_anticipated_fishing_entries(&ctx, &stage, 0, &layout, &mut col_entries);

        let block_hours_total: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
        let expected_neg = -block_hours_total;
        for local_idx in 0..layout.anticipated.n_anticipated_fishing_rows {
            let row = layout.anticipated.row_anticipated_fishing_start + local_idx;
            let col_state = layout.col_anticipated_state_start() + local_idx;
            let state_couplings: Vec<&(usize, f64)> = col_entries[col_state]
                .iter()
                .filter(|(r, _)| *r == row)
                .collect();
            assert_eq!(
                state_couplings.len(),
                1,
                "anticipated_state col {col_state} must carry exactly 1 coupling \
                 at fishing row {row} (plant local_idx={local_idx})"
            );
            let (_, coeff) = state_couplings[0];
            assert!(
                (coeff - expected_neg).abs() < 1e-12,
                "anticipated_state col {col_state} fishing-row coupling: \
                 expected {expected_neg}, got {coeff} (plant local_idx={local_idx})"
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Fixture helper for anticipated state-out tests
    // ─────────────────────────────────────────────────────────────────────────

    /// Build a minimal fixture for anticipated state-out tests:
    /// `n_anticipated = 2`, `K = [2, 3]`, `n_stages = 6`.
    ///
    /// `ResolvedBounds` is constructed with the correct `n_stages` so that
    /// `ctx.resolved.bounds.n_stages()` returns 6, which is required by
    /// `fill_anticipated_state_out_columns`, `fill_anticipated_state_out_def_rows`,
    /// and `fill_anticipated_state_out_def_entries`.
    fn build_anticipated_ctx_n_stages_6() -> (AntFixtures, Stage) {
        let mut fixtures = AntFixtures::new();
        // Override bounds with a 6-stage table (zero entities, k_max = 3).
        fixtures.bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: 6,
                k_max: 3,
            },
            &BoundsDefaults {
                hydro: HydroStageBounds {
                    min_storage_hm3: 0.0,
                    max_storage_hm3: 0.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 0.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 0.0,
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
        );
        let stage = two_block_stage(0);
        (fixtures, stage)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tests for fill_anticipated_state_out_columns
    // ─────────────────────────────────────────────────────────────────────────

    /// Active plants (`stage_idx + K_i < n_stages`) get `[-INF, +INF]` bounds;
    /// inactive plants (`stage_idx + K_i >= n_stages`) get `[0, 0]` bounds.
    ///
    /// Fixture: `n_anticipated=2`, `K=[2, 3]`, `n_stages=6`.
    /// Stage 0: both plants active  (0+2=2 < 6, 0+3=3 < 6) → `[-INF, +INF]`.
    /// Stage 5: both plants inactive (5+2=7 >= 6, 5+3=8 >= 6) → `[0, 0]`.
    #[test]
    fn test_fill_anticipated_state_out_columns_active_and_inactive() {
        let (fixtures, _) = build_anticipated_ctx_n_stages_6();
        let ctx = fixtures.make_ctx(
            2,          // n_anticipated
            3,          // k_max
            vec![2, 3], // anticipated_lead_stages: K=[2,3]
            vec![0, 1], // anticipated_thermal_indices
            0,          // n_thermals
        );

        // Stage 0: both plants active.
        let stage0 = two_block_stage(0);
        let layout0 = StageLayout::new(&ctx, &stage0, 0);
        let mut col_lower = vec![0.0_f64; layout0.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout0.num_cols];
        let mut objective = vec![0.0_f64; layout0.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_anticipated_state_out_columns(&ctx, 0, &layout0, &mut bufs);
        for i in 0..2 {
            let col = layout0.anticipated.col_anticipated_state_out_start + i;
            assert_eq!(
                col_lower[col],
                f64::NEG_INFINITY,
                "stage 0, plant {i}: col_lower expected -INF, got {}",
                col_lower[col]
            );
            assert_eq!(
                col_upper[col],
                f64::INFINITY,
                "stage 0, plant {i}: col_upper expected +INF, got {}",
                col_upper[col]
            );
        }

        // Stage 5: both plants inactive.
        let stage5 = two_block_stage(5);
        let layout5 = StageLayout::new(&ctx, &stage5, 5);
        let mut col_lower5 = vec![0.0_f64; layout5.num_cols];
        let mut col_upper5 = vec![f64::INFINITY; layout5.num_cols];
        let mut objective5 = vec![0.0_f64; layout5.num_cols];
        let mut bufs5 = ColumnBufs {
            col_lower: &mut col_lower5,
            col_upper: &mut col_upper5,
            objective: &mut objective5,
        };
        fill_anticipated_state_out_columns(&ctx, 5, &layout5, &mut bufs5);
        assert_eq!(
            layout5.anticipated.n_anticipated_state_out_def_rows, 0,
            "stage 5 inactive: expected no def rows, got {}",
            layout5.anticipated.n_anticipated_state_out_def_rows,
        );
        for i in 0..2 {
            let col = layout5.anticipated.col_anticipated_state_out_start + i;
            assert_eq!(
                col_lower5[col], 0.0,
                "stage 5, plant {i}: col_lower expected 0.0, got {}",
                col_lower5[col]
            );
            assert_eq!(
                col_upper5[col], 0.0,
                "stage 5, plant {i}: col_upper expected 0.0, got {}",
                col_upper5[col]
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tests for fill_anticipated_state_out_def_rows
    // ─────────────────────────────────────────────────────────────────────────

    /// At stage 0 with `K=[2,3]` and `n_stages=6`, both plants are active
    /// (0+2 < 6, 0+3 < 6), so `n_anticipated_state_out_def_rows == 2` and
    /// both definition rows must have equality bounds `[0.0, 0.0]`.
    #[test]
    fn test_fill_anticipated_state_out_def_rows_two_active_plants() {
        let (fixtures, stage) = build_anticipated_ctx_n_stages_6();
        let ctx = fixtures.make_ctx(2, 3, vec![2, 3], vec![0, 1], 0);
        let layout = StageLayout::new(&ctx, &stage, 0);

        assert_eq!(
            layout.anticipated.n_anticipated_state_out_def_rows, 2,
            "expected n_anticipated_state_out_def_rows == 2, got {}",
            layout.anticipated.n_anticipated_state_out_def_rows
        );

        let mut row_lower = vec![f64::NEG_INFINITY; layout.num_rows];
        let mut row_upper = vec![f64::INFINITY; layout.num_rows];
        fill_anticipated_state_out_def_rows(&ctx, 0, &layout, &mut row_lower, &mut row_upper);

        for k in 0..2 {
            let row = layout.anticipated.row_anticipated_state_out_def_start + k;
            assert_eq!(
                row_lower[row], 0.0,
                "def row {k}: row_lower expected 0.0, got {}",
                row_lower[row]
            );
            assert_eq!(
                row_upper[row], 0.0,
                "def row {k}: row_upper expected 0.0, got {}",
                row_upper[row]
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tests for fill_anticipated_state_out_def_entries
    // ─────────────────────────────────────────────────────────────────────────

    /// At stage 0 with `K=[2,3]` and `n_stages=6`, both plants are active.
    /// For each active plant `i`, the CSC entry list must contain:
    /// - `(def_row_i, +1.0)` on `col_anticipated_state_out_start + i`
    /// - `(def_row_i, -1.0)` on `col_anticipated_decision_start + i`
    #[test]
    fn test_fill_anticipated_state_out_def_entries_two_active_plants() {
        let (fixtures, stage) = build_anticipated_ctx_n_stages_6();
        let ctx = fixtures.make_ctx(2, 3, vec![2, 3], vec![0, 1], 0);
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_anticipated_state_out_def_entries(&ctx, 0, &layout, &mut col_entries);

        for k in 0..2 {
            let row = layout.anticipated.row_anticipated_state_out_def_start + k;
            let col_state_out = layout.anticipated.col_anticipated_state_out_start + k;
            let col_decision = layout.anticipated.col_anticipated_decision_start + k;

            assert!(
                col_entries[col_state_out]
                    .iter()
                    .any(|&(r, v)| r == row && (v - 1.0).abs() < 1e-15),
                "plant {k}: expected (+1.0) entry at (col_state_out={col_state_out}, row={row}), \
                 got {:?}",
                col_entries[col_state_out]
            );
            assert!(
                col_entries[col_decision]
                    .iter()
                    .any(|&(r, v)| r == row && (v + 1.0).abs() < 1e-15),
                "plant {k}: expected (-1.0) entry at (col_decision={col_decision}, row={row}), \
                 got {:?}",
                col_entries[col_decision]
            );
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // State-fixing diagonals must be absent from the CSC output
    // ─────────────────────────────────────────────────────────────────────────

    /// Asserts `build_stage_matrix_entries` produces no state-fixing
    /// diagonals in the CSC output.
    ///
    /// Coverage strategy: storage-fixing and lag-fixing diagonals are
    /// guaranteed absent by structural deletion of their for-loops in
    /// `fill_state_and_water_entries` (verified by C1+C2 grep — the
    /// functions/loops emitting those entries no longer exist in the
    /// source). Anticipated-state-fixing diagonals are checked dynamically:
    /// the test builds a fixture with `n_anticipated = 2, k_max = 3` and
    /// asserts every `(slot, plant)` column at
    /// `col_anticipated_state_start + slot*A + plant` has no entry at
    /// row `slot*A + plant` (the diagonal entry that existed in the
    /// pre-cutover layout, before state pinning moved to column bounds).
    ///
    /// The storage/lag assertions are included as zero-iteration loops in
    /// this fixture (`n_hydros` = 0) so the test documents the intent and
    /// would catch a future regression in any fixture that adds hydros.
    #[test]
    fn state_fixing_diagonals_absent_from_csc() {
        let (fixtures, stage) = build_anticipated_ctx_n_stages_6();
        let ctx = fixtures.make_ctx(
            2,          // n_anticipated
            3,          // k_max
            vec![2, 3], // anticipated_lead_stages
            vec![0, 1], // anticipated_thermal_indices
            0,          // n_thermals
        );

        let layout = StageLayout::new(&ctx, &stage, 0);
        let col_entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);

        let a = ctx.n_anticipated;
        let k = ctx.k_max;
        for slot in 0..k {
            for plant in 0..a {
                let col = layout.col_anticipated_state_start() + slot * a + plant;
                let diag_row = slot * a + plant;
                let has_diag = col_entries[col]
                    .iter()
                    .any(|&(r, v)| r == diag_row && (v - 1.0).abs() < 1e-15);
                assert!(
                    !has_diag,
                    "anticipated-state-fixing diagonal (row={diag_row}, val=1.0) must be absent \
                     from col {col} (slot={slot}, plant={plant})"
                );
            }
        }

        // Storage-fixing and lag-fixing diagonal absence assertions. With
        // n_hydros = 0 in this fixture these loops execute zero iterations,
        // but the structure documents intent and the same assertion shape
        // catches a regression in any future fixture with non-zero hydros.
        let n_h = ctx.n_hydros;
        let lag_order = ctx.max_par_order;
        for h in 0..n_h {
            let col = layout.col_storage_in_start() + h;
            let has_diag = col_entries[col]
                .iter()
                .any(|&(r, v)| r == h && (v - 1.0).abs() < 1e-15);
            assert!(
                !has_diag,
                "storage-fixing diagonal (row={h}, val=1.0) must be absent from col {col}"
            );
        }
        for lag in 0..lag_order {
            for h in 0..n_h {
                let col = layout.col_inflow_lags_start() + lag * n_h + h;
                let diag_row = n_h + lag * n_h + h;
                let has_diag = col_entries[col]
                    .iter()
                    .any(|&(r, v)| r == diag_row && (v - 1.0).abs() < 1e-15);
                assert!(
                    !has_diag,
                    "lag-fixing diagonal (row={diag_row}, val=1.0) must be absent from col {col} \
                     (lag={lag}, h={h})"
                );
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod pumping_water_tests {
    use std::collections::HashMap;

    use chrono::NaiveDate;
    use cobre_core::{
        Block, BlockMode, BoundsCountsSpec, BoundsDefaults, Bus, CascadeTopology, CoefficientRef,
        ConstraintExpression, ConstraintSense, ContractStageBounds, DeficitSegment, EntityId,
        GenericConstraint, Hydro, HydroGenerationModel, HydroPenalties, HydroStageBounds,
        LineStageBounds, LinearTerm, NoiseMethod, PumpingStageBounds, PumpingStation,
        ResolvedBounds, ResolvedExchangeFactors, ResolvedGenericConstraintBounds,
        ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties,
        ScenarioSourceConfig, SlackConfig, Stage, StageRiskConfig, StageStateConfig,
        ThermalStageBounds, VariableRef,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{EvaporationModel, EvaporationModelSet, ProductionModelSet};
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::{
        ColumnBufs, LpMatrixBuffers, M3S_TO_HM3, StageLayout, TemplateBuildCtx, assemble_csc,
        build_stage_matrix_entries, fill_generic_constraint_entries, fill_load_balance_entries,
        fill_pumping_columns, fill_pumping_water_entries,
    };

    const N_STAGES: usize = 1;

    /// Minimal independent (no-downstream) constant-productivity hydro.
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

    fn zero_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
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

    fn default_bounds_defaults() -> BoundsDefaults {
        BoundsDefaults {
            hydro: HydroStageBounds {
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
        }
    }

    /// Build a two-block `Stage` with distinct block durations so a τ that
    /// confuses the block index is observable.
    fn two_block_stage() -> Stage {
        Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![
                Block {
                    index: 0,
                    name: "BLK0".to_string(),
                    duration_hours: 300.0,
                },
                Block {
                    index: 1,
                    name: "BLK1".to_string(),
                    duration_hours: 444.0,
                },
            ],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    /// A single bus with one unbounded deficit segment, on `EntityId(1)` (the bus
    /// the fixture hydros and `station` helper reference).
    fn fixture_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("B{id}"),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 1.0,
        }
    }

    /// A pumping station on `EntityId(1)` with explicit source/destination/flow
    /// data and the default `consumption_mw_per_m3s` (0.5).
    fn station(
        id: i32,
        source: i32,
        destination: i32,
        min_flow: f64,
        max_flow: f64,
    ) -> PumpingStation {
        station_full(id, source, destination, min_flow, max_flow, 1, 0.5)
    }

    /// A pumping station with an explicit `bus_id` and consumption rate so the
    /// power-coupling tests can place a station on an unmapped bus and observe a
    /// distinct coefficient.
    fn station_full(
        id: i32,
        source: i32,
        destination: i32,
        min_flow: f64,
        max_flow: f64,
        bus_id: i32,
        consumption_mw_per_m3s: f64,
    ) -> PumpingStation {
        PumpingStation {
            id: EntityId(id),
            name: format!("P{id}"),
            bus_id: EntityId(bus_id),
            source_hydro_id: EntityId(source),
            destination_hydro_id: EntityId(destination),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s,
            min_flow_m3s: min_flow,
            max_flow_m3s: max_flow,
        }
    }

    /// Owns the data backing a two-hydro `TemplateBuildCtx` carrying pumping
    /// stations. Hydros and stations are stored in canonical (ID-sorted) order;
    /// `hydro_pos`/`pumping_pos` are derived from those sorted slices, exactly as
    /// `SystemBuilder::build` produces them in production.
    struct PumpFixtures {
        hydros: Vec<Hydro>,
        stations: Vec<PumpingStation>,
        buses: Vec<Bus>,
        hydro_pos: HashMap<EntityId, usize>,
        pumping_pos: HashMap<EntityId, usize>,
        bus_pos: HashMap<EntityId, usize>,
        par_lp: PrecomputedPar,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_exchange_factors: ResolvedExchangeFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        /// Generic constraints whose expressions the LP builder resolves. Empty by
        /// default; the end-to-end test sets a pumping-referencing constraint so the
        /// `PumpingFlow`/`PumpingPower` resolver arms run through the real caller.
        generic_constraints: Vec<GenericConstraint>,
    }

    impl PumpFixtures {
        /// Build a fixture with a single bus (`EntityId(1)`) — the bus the fixture
        /// hydros and the `station` helper reference — so the load-balance row is
        /// present and the pumping-power coupling is exercised.
        fn new(hydros: Vec<Hydro>, stations: Vec<PumpingStation>) -> Self {
            Self::new_with_buses(hydros, stations, vec![fixture_bus(1)])
        }

        /// Build a fixture from hydros, stations, and buses supplied in arbitrary
        /// declaration order. All three are sorted by `id.0` (the canonical
        /// operation `SystemBuilder::build` performs) before deriving position maps
        /// and the pumping bounds table, so the resulting ctx is
        /// declaration-order-invariant.
        fn new_with_buses(
            mut hydros: Vec<Hydro>,
            mut stations: Vec<PumpingStation>,
            mut buses: Vec<Bus>,
        ) -> Self {
            hydros.sort_by_key(|h| h.id.0);
            stations.sort_by_key(|s| s.id.0);
            buses.sort_by_key(|b| b.id.0);

            let hydro_pos: HashMap<EntityId, usize> =
                hydros.iter().enumerate().map(|(i, h)| (h.id, i)).collect();
            let pumping_pos: HashMap<EntityId, usize> = stations
                .iter()
                .enumerate()
                .map(|(i, s)| (s.id, i))
                .collect();
            let bus_pos: HashMap<EntityId, usize> =
                buses.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

            let mut bounds = ResolvedBounds::new(
                &BoundsCountsSpec {
                    n_hydros: hydros.len(),
                    n_thermals: 0,
                    n_lines: 0,
                    n_pumping: stations.len(),
                    n_contracts: 0,
                    n_stages: N_STAGES,
                    k_max: 0,
                },
                &default_bounds_defaults(),
            );
            // Distinct per-station bounds so a column/bound mismatch is observable.
            for (p_idx, s) in stations.iter().enumerate() {
                for stage_idx in 0..N_STAGES {
                    *bounds.pumping_bounds_mut(p_idx, stage_idx) = PumpingStageBounds {
                        min_flow_m3s: s.min_flow_m3s,
                        max_flow_m3s: s.max_flow_m3s,
                    };
                }
            }

            let production_models = ProductionModelSet::new(
                vec![
                    vec![
                        crate::hydro_models::ResolvedProductionModel::ConstantProductivity {
                            productivity: 1.0,
                        };
                        N_STAGES
                    ];
                    hydros.len()
                ],
                hydros.len(),
                N_STAGES,
            );
            let evaporation_models =
                EvaporationModelSet::new(vec![EvaporationModel::None; hydros.len()]);

            Self {
                hydros,
                stations,
                buses,
                hydro_pos,
                pumping_pos,
                bus_pos,
                par_lp: PrecomputedPar::default(),
                cascade: CascadeTopology::build(&[]),
                bounds,
                penalties: ResolvedPenalties::empty(),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_exchange_factors: ResolvedExchangeFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                },
                production_models,
                evaporation_models,
                generic_constraints: Vec::new(),
            }
        }

        /// Attach a generic constraint (and its active-at-stage-0 bound) so the
        /// LP builder resolves the constraint's expression against the pumping
        /// columns. Used by the end-to-end resolver-integration test.
        fn with_generic_constraint(mut self, constraint: GenericConstraint, bound: f64) -> Self {
            let constraint_id = constraint.id.0;
            let id_map: HashMap<i32, usize> = [(constraint_id, 0)].into_iter().collect();
            let rows = (0..N_STAGES).map(|s| {
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                (constraint_id, s as i32, None, bound)
            });
            self.resolved_generic_bounds =
                ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());
            self.generic_constraints = vec![constraint];
            self
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            TemplateBuildCtx {
                hydros: &self.hydros,
                thermals: &[],
                lines: &[],
                buses: &self.buses,
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
                hydro_pos: self.hydro_pos.clone(),
                thermal_pos: HashMap::new(),
                line_pos: HashMap::new(),
                bus_pos: self.bus_pos.clone(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &self.generic_constraints,
                non_controllable_sources: &[],
                pumping_stations: &self.stations,
                pumping_pos: self.pumping_pos.clone(),
                n_pumping: self.stations.len(),
                diversion_upstream: HashMap::new(),
                n_hydros: self.hydros.len(),
                n_thermals: 0,
                n_lines: 0,
                n_buses: self.buses.len(),
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                has_penalty: false,
                cumulative_discount_factors: vec![1.0; N_STAGES],
                total_hours_per_stage: vec![744.0; N_STAGES],
            }
        }
    }

    /// Column bounds = `[min_flow, max_flow]` and zero objective for every
    /// `(station, block)` pumping column.
    #[test]
    fn pumping_columns_get_flow_bounds_and_zero_cost() {
        let stations = vec![station(10, 1, 2, 5.0, 80.0), station(20, 2, 1, 0.0, 30.0)];
        let fixtures = PumpFixtures::new(vec![fixture_hydro(1), fixture_hydro(2)], stations);
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);

        // Lower/upper start at a NaN sentinel so any column the helper fails to
        // bound is visible; objective starts at the production default (0.0), so
        // the post-fill zero assertion proves the helper writes no cost.
        let mut col_lower = vec![f64::NAN; layout.num_cols];
        let mut col_upper = vec![f64::NAN; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };

        fill_pumping_columns(&ctx, 0, &layout, &mut bufs);

        let n_blks = layout.n_blks;
        for (p_idx, s) in ctx.pumping_stations.iter().enumerate() {
            for blk in 0..n_blks {
                let col = layout.col_pumping_start + p_idx * n_blks + blk;
                assert_eq!(
                    bufs.col_lower[col], s.min_flow_m3s,
                    "station {p_idx} blk {blk}: lower bound must be min_flow"
                );
                assert_eq!(
                    bufs.col_upper[col], s.max_flow_m3s,
                    "station {p_idx} blk {blk}: upper bound must be max_flow"
                );
                assert_eq!(
                    bufs.objective[col], 0.0,
                    "station {p_idx} blk {blk}: objective must be zero"
                );
            }
        }
    }

    /// Source water row gains `+tau_h`, destination water row gains `−tau_h`,
    /// with `tau_h == block.duration_hours * M3S_TO_HM3` per block.
    #[test]
    fn pumping_water_entries_source_plus_tau_destination_minus_tau() {
        // Station id 10: source hydro id 1 (pos 0), destination hydro id 2 (pos 1).
        let fixtures = PumpFixtures::new(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station(10, 1, 2, 0.0, 50.0)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_pumping_water_entries(&ctx, &stage, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        let source_pos = ctx.hydro_pos[&EntityId(1)];
        let dest_pos = ctx.hydro_pos[&EntityId(2)];
        let row_source = layout.row_water_balance_start() + source_pos;
        let row_dest = layout.row_water_balance_start() + dest_pos;

        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col = layout.col_pumping_start + blk;
            assert_eq!(
                col_entries[col],
                vec![(row_source, tau_h), (row_dest, -tau_h)],
                "blk {blk}: source +tau_h then destination -tau_h"
            );
        }
    }

    /// A station whose `source_hydro_id` is absent from `hydro_pos` skips only
    /// the source entry — the destination side is still written, no panic.
    #[test]
    fn pumping_water_entries_missing_source_skips_only_source() {
        // Source hydro id 99 does NOT exist; destination hydro id 2 (pos 1) does.
        let fixtures = PumpFixtures::new(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station(10, 99, 2, 0.0, 50.0)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_pumping_water_entries(&ctx, &stage, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        let dest_pos = ctx.hydro_pos[&EntityId(2)];
        let row_dest = layout.row_water_balance_start() + dest_pos;
        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col = layout.col_pumping_start + blk;
            assert_eq!(
                col_entries[col],
                vec![(row_dest, -tau_h)],
                "blk {blk}: only the destination -tau_h entry survives"
            );
        }
    }

    /// A station whose `destination_hydro_id` is absent skips only the
    /// destination entry — the source side is still written, no panic.
    #[test]
    fn pumping_water_entries_missing_destination_skips_only_destination() {
        // Source hydro id 1 (pos 0) exists; destination hydro id 99 does NOT.
        let fixtures = PumpFixtures::new(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station(10, 1, 99, 0.0, 50.0)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_pumping_water_entries(&ctx, &stage, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        let source_pos = ctx.hydro_pos[&EntityId(1)];
        let row_source = layout.row_water_balance_start() + source_pos;
        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col = layout.col_pumping_start + blk;
            assert_eq!(
                col_entries[col],
                vec![(row_source, tau_h)],
                "blk {blk}: only the source +tau_h entry survives"
            );
        }
    }

    /// The pumping flow column enters its bus load-balance row with
    /// `−consumption_mw_per_m3s` per block — a negative injection, the same sign a
    /// line carries into its source bus, NOT the `+1.0` of generation.
    #[test]
    fn pumping_power_enters_bus_row_with_negative_consumption() {
        // Station id 10 on bus id 1 (pos 0), consumption 0.75 MW per m³/s.
        let fixtures = PumpFixtures::new_with_buses(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station_full(10, 1, 2, 0.0, 50.0, 1, 0.75)],
            vec![fixture_bus(1)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        let b_idx = ctx.bus_pos[&EntityId(1)];
        for blk in 0..n_blks {
            let row = layout.row_load_balance_start() + b_idx * n_blks + blk;
            let col = layout.col_pumping_start + blk;
            assert!(
                col_entries[col].contains(&(row, -0.75)),
                "blk {blk}: pumping column {col} must carry (row {row}, -0.75); got {:?}",
                col_entries[col]
            );
            // The flow column carries the bus-power coupling and nothing else from
            // the load-balance fill (no positive generation-style entry).
            assert_eq!(
                col_entries[col],
                vec![(row, -0.75)],
                "blk {blk}: pumping column must carry only the negative-injection entry"
            );
        }
    }

    /// A station whose `bus_id` is absent from `bus_pos` writes no load-balance
    /// entry and does not panic.
    #[test]
    fn pumping_power_missing_bus_skips_without_panic() {
        // Station on bus id 99, which is NOT among the fixture buses (only id 1).
        let fixtures = PumpFixtures::new_with_buses(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station_full(10, 1, 2, 0.0, 50.0, 99, 0.5)],
            vec![fixture_bus(1)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        for blk in 0..n_blks {
            let col = layout.col_pumping_start + blk;
            assert!(
                col_entries[col].is_empty(),
                "blk {blk}: station on an unmapped bus must write no load-balance entry"
            );
        }
    }

    /// With no pumping stations, `fill_load_balance_entries` produces exactly the
    /// same entries it would without the pumping loop — the pumping path is inert.
    /// A 1-bus, 2-hydro system with one declared station is the baseline; removing
    /// the station must leave every column's load-balance entries unchanged.
    #[test]
    fn no_pumping_stations_leaves_load_balance_entries_identical() {
        let build = |stations: Vec<PumpingStation>| {
            let fixtures = PumpFixtures::new_with_buses(
                vec![fixture_hydro(1), fixture_hydro(2)],
                stations,
                vec![fixture_bus(1)],
            );
            let ctx = fixtures.make_ctx();
            let stage = two_block_stage();
            let layout = StageLayout::new(&ctx, &stage, 0);
            let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
            fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);
            // Truncate to the non-pumping column region: with zero stations the
            // layout has no pumping columns, so compare the shared prefix that both
            // layouts share (generation/thermal/line/deficit/excess columns are all
            // indexed before the pumping block).
            (layout.col_pumping_start, col_entries)
        };

        let (pump_start_empty, entries_empty) = build(vec![]);
        let (_pump_start_one, entries_one) = build(vec![station(10, 1, 2, 0.0, 50.0)]);

        // Every column before the pumping block must carry identical load-balance
        // entries whether or not a station is present.
        for col in 0..pump_start_empty {
            assert_eq!(
                entries_empty[col], entries_one[col],
                "load-balance entries for column {col} must be pumping-independent"
            );
        }
    }

    /// Build the full CSC for a 2-reservoir + 1-bus system twice with the hydro,
    /// station, and bus declarations supplied in two DIFFERENT input orders, and
    /// assert the assembled CSC arrays are byte-identical. Determinism is a hard
    /// rule: the canonical ID-sort plus the per-column row-sort must erase all
    /// trace of the input declaration order.
    ///
    /// Two stations with opposite source/destination orientation are declared so
    /// the assertion is load-bearing on the pumping path: a single station would
    /// pass even if the pumping iteration were declaration-order-dependent (there
    /// is nothing to scramble), whereas permuting two stations exercises the
    /// per-column row-sort that decouples declaration order from CSC layout. Both
    /// stations sit on the single bus, so the `−consumption_mw_per_m3s` bus-power
    /// entries are part of the assembled CSC and the assertion covers them too.
    #[test]
    fn csc_byte_identical_under_permuted_declaration_order() {
        let assemble = |hydros: Vec<Hydro>, stations: Vec<PumpingStation>| {
            let fixtures = PumpFixtures::new(hydros, stations);
            let ctx = fixtures.make_ctx();
            let stage = two_block_stage();
            let layout = StageLayout::new(&ctx, &stage, 0);
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            // Mirror the production per-column row-sort (see build_single_stage_template).
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };

        // Two reservoirs (ids 1, 2) and two stations moving water in opposite
        // directions (10: 1 → 2, 20: 2 → 1), both on bus 1 with DISTINCT
        // consumption rates so a permutation that mislabels which station's
        // `−consumption_mw_per_m3s` lands on which pumping column would be caught.
        // Order A declares both ascending.
        let csc_a = assemble(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![
                station_full(10, 1, 2, 5.0, 80.0, 1, 0.4),
                station_full(20, 2, 1, 0.0, 30.0, 1, 0.9),
            ],
        );
        // Order B declares the identical entities with both hydros and both
        // stations reversed.
        let csc_b = assemble(
            vec![fixture_hydro(2), fixture_hydro(1)],
            vec![
                station_full(20, 2, 1, 0.0, 30.0, 1, 0.9),
                station_full(10, 1, 2, 5.0, 80.0, 1, 0.4),
            ],
        );

        assert_eq!(csc_a.0, csc_b.0, "col_starts must be byte-identical");
        assert_eq!(csc_a.1, csc_b.1, "row_indices must be byte-identical");
        assert_eq!(csc_a.2, csc_b.2, "values must be byte-identical");
    }

    /// End-to-end: a generic constraint referencing `pumping_flow` and
    /// `pumping_power` resolves to the REAL pumping column(s) through the
    /// resolver's sole caller (`fill_generic_constraint_entries`), and the
    /// constraint participates in the LP — its row carries CSC entries on the
    /// pumping columns. The `block_id = None` expression is block-dependent, so it
    /// expands to one generic row per block; each row's two terms (flow ×1.0 and
    /// power ×consumption) alias the SAME pumping column for that block, so the
    /// summed coefficient at `(pumping_col, generic_row)` is `1.0 + consumption`.
    #[test]
    fn b6b_generic_constraint_resolves_pumping_columns_in_lp() {
        let consumption = 0.5_f64;
        let constraint_id = EntityId(7);
        let station_id = EntityId(10);

        // pumping_flow(10) + pumping_power(10) <= 40 (block_id = None on both).
        let constraint = GenericConstraint {
            id: constraint_id,
            name: "gc_pump".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![
                    LinearTerm {
                        coefficient: CoefficientRef::Literal(1.0),
                        scale: 1.0,
                        variable: VariableRef::PumpingFlow {
                            station_id,
                            block_id: None,
                        },
                    },
                    LinearTerm {
                        coefficient: CoefficientRef::Literal(1.0),
                        scale: 1.0,
                        variable: VariableRef::PumpingPower {
                            station_id,
                            block_id: None,
                        },
                    },
                ],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };

        let fixtures = PumpFixtures::new(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![station_full(station_id.0, 1, 2, 0.0, 50.0, 1, consumption)],
        )
        .with_generic_constraint(constraint, 40.0);
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);

        // Block-dependent expression with block_id = None expands to one generic
        // row per block, so the constraint participates as `n_blks` rows.
        let n_blks = layout.n_blks;
        assert_eq!(
            layout.n_generic_rows, n_blks,
            "block-dependent pumping constraint must expand to one row per block"
        );

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut row_lower = vec![f64::NEG_INFINITY; layout.num_rows];
        let mut row_upper = vec![f64::INFINITY; layout.num_rows];
        let mut buffers = LpMatrixBuffers {
            col_entries: &mut col_entries,
            col_upper: &mut col_upper,
            objective: &mut objective,
            row_lower: &mut row_lower,
            row_upper: &mut row_upper,
        };

        fill_generic_constraint_entries(&ctx, &stage, 0, &layout, &mut buffers);

        // Each generic row `blk` lands on the station's flow column for that block,
        // with the flow (1.0) and power (consumption) terms aliasing the SAME column.
        // p_idx = 0 (the only station), so col = col_pumping_start + blk.
        for blk in 0..n_blks {
            let row = layout.row_generic_start + blk;
            let col = layout.col_pumping_start + blk;
            let summed: f64 = col_entries[col]
                .iter()
                .filter(|&&(r, _)| r == row)
                .map(|&(_, v)| v)
                .sum();
            assert_eq!(
                summed,
                1.0 + consumption,
                "blk {blk}: pumping column {col} must carry flow(1.0) + power({consumption}) on generic row {row}"
            );
            // The row bound proves the constraint participates with the right sense.
            assert_eq!(row_upper[row], 40.0, "blk {blk}: <= row upper bound");
            assert_eq!(row_lower[row], f64::NEG_INFINITY, "blk {blk}: <= row lower");
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod diversion_bound_tests {
    use std::collections::HashMap;

    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{DiversionChannel, HydroGenerationModel, HydroPenalties};
    use cobre_core::{
        Block, BlockMode, BoundsCountsSpec, BoundsDefaults, CascadeTopology, ContractStageBounds,
        EntityId, Hydro, HydroStageBounds, HydroStagePenalties, LineStageBounds, NoiseMethod,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds, ResolvedBounds,
        ResolvedExchangeFactors, ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors,
        ResolvedPenalties, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
        ThermalStageBounds,
    };
    use cobre_core::{BusStagePenalties, LineStagePenalties, NcsStagePenalties};
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, ProductionModelSet, ResolvedProductionModel,
    };
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::{ColumnBufs, StageLayout, TemplateBuildCtx, fill_diversion_columns};

    // Declaration-time diversion capacity on the entity. The test makes the
    // resolved per-stage override distinct from this so the two reads disagree:
    // reading the entity directly (the pre-fix bug) would yield this value.
    const DECLARATION_MAX_FLOW_M3S: f64 = 200.0;
    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;

    fn two_block_stage() -> Stage {
        Stage {
            index: STAGE_IDX,
            id: STAGE_IDX as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![
                Block {
                    index: 0,
                    name: "BLK0".to_string(),
                    duration_hours: 372.0,
                },
                Block {
                    index: 1,
                    name: "BLK1".to_string(),
                    duration_hours: 372.0,
                },
            ],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn zero_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
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
            let mut hydro_pos = HashMap::new();
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
                thermal_pos: HashMap::new(),
                line_pos: HashMap::new(),
                bus_pos: HashMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: HashMap::new(),
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
        let stage = two_block_stage();
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
    use std::collections::HashMap;

    use chrono::NaiveDate;
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::{
        Block, BlockMode, BoundsCountsSpec, BoundsDefaults, BusStagePenalties, CascadeTopology,
        ContractStageBounds, EntityId, Hydro, HydroStageBounds, HydroStagePenalties,
        LineStageBounds, LineStagePenalties, NcsStagePenalties, NoiseMethod, PenaltiesCountsSpec,
        PenaltiesDefaults, PumpingStageBounds, ResolvedBounds, ResolvedExchangeFactors,
        ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties,
        ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, ProductionModelSet, ResolvedProductionModel,
    };
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
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

    fn zero_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
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

    /// Two-block stage with distinct durations so a block-index confusion in the
    /// objective scaling is observable.
    fn two_block_stage() -> Stage {
        Stage {
            index: STAGE_IDX,
            id: STAGE_IDX as i32,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![
                Block {
                    index: 0,
                    name: "BLK0".to_string(),
                    duration_hours: BLOCK_HOURS[0],
                },
                Block {
                    index: 1,
                    name: "BLK1".to_string(),
                    duration_hours: BLOCK_HOURS[1],
                },
            ],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
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
            let mut hydro_pos = HashMap::new();
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
                thermal_pos: HashMap::new(),
                line_pos: HashMap::new(),
                bus_pos: HashMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                non_controllable_sources: &[],
                pumping_stations: &[],
                pumping_pos: HashMap::new(),
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
        let stage = two_block_stage();
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
