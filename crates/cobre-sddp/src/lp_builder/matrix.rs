use cobre_core::{CoefficientRef, ConstraintSense, Stage};

use crate::generic_constraints::resolve_variable_ref;
use crate::hydro_models::{EvaporationModel, ResolvedProductionModel};
use crate::indexer::StageIndexer;

use super::layout::{StageLayout, TemplateBuildCtx};
use super::{M3S_TO_HM3, Q_EV_SAFETY_MARGIN};

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
    let idx = StageIndexer::new(ctx.n_hydros, ctx.max_par_order);
    let mut col_lower = vec![0.0_f64; layout.num_cols];
    let mut col_upper = vec![f64::INFINITY; layout.num_cols];
    let mut objective = vec![0.0_f64; layout.num_cols];
    let total_stage_hours: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
    let b = &mut ColumnBufs {
        col_lower: &mut col_lower,
        col_upper: &mut col_upper,
        objective: &mut objective,
    };

    fill_storage_columns(ctx, stage_idx, &idx, layout, b);
    fill_ar_lag_columns(&idx, b);
    fill_theta_column(&idx, b);
    fill_turbine_columns(ctx, stage, stage_idx, layout, b);
    fill_spillage_columns(ctx, stage, stage_idx, layout, b);
    fill_diversion_columns(ctx, stage, stage_idx, layout, b);
    fill_thermal_columns(ctx, stage, stage_idx, layout, b);
    fill_line_columns(ctx, stage, stage_idx, layout, b);
    fill_deficit_and_excess_columns(ctx, stage, stage_idx, layout, b);
    fill_inflow_slack_columns(ctx, stage_idx, layout, total_stage_hours, b);
    fill_fpha_generation_columns(ctx, stage_idx, layout, b);
    fill_evaporation_columns(ctx, stage_idx, layout, total_stage_hours, b);
    fill_withdrawal_slack_columns(ctx, stage_idx, layout, total_stage_hours, b);
    fill_operational_slack_columns(ctx, stage, stage_idx, layout, b);
    fill_ncs_columns(ctx, stage, stage_idx, layout, b);
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
    idx: &StageIndexer,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        bufs.col_lower[h_idx] = hb.min_storage_hm3;
        bufs.col_upper[h_idx] = hb.max_storage_hm3;
        bufs.col_lower[idx.storage_in.start + h_idx] = f64::NEG_INFINITY;
        bufs.col_upper[idx.storage_in.start + h_idx] = f64::INFINITY;
    }
}

/// AR lag columns: unconstrained (signed).
fn fill_ar_lag_columns(idx: &StageIndexer, bufs: &mut ColumnBufs<'_>) {
    for lag_col in idx.inflow_lags.clone() {
        bufs.col_lower[lag_col] = f64::NEG_INFINITY;
        bufs.col_upper[lag_col] = f64::INFINITY;
    }
}

/// Theta column: bounded below by zero so iteration-1 LPs with empty cut pools
/// are bounded rather than unbounded.
fn fill_theta_column(idx: &StageIndexer, bufs: &mut ColumnBufs<'_>) {
    bufs.col_lower[idx.theta] = 0.0;
    bufs.col_upper[idx.theta] = f64::INFINITY;
    bufs.objective[idx.theta] = 1.0;
}

/// Turbine columns per hydro per block.
///
/// For constant-productivity hydros, caps turbine flow so that
/// `productivity * turbined <= max_generation_mw` (derated capacity).
/// Carries `turbined_cost * block_hours` in the objective on every hydro's
/// turbine column regardless of production model — matches NEWAVE behavior
/// where the turbined cost applies to every plant, not only FPHA hydros.
fn fill_turbine_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
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
            let col = layout.col_turbine_start + h_idx * layout.n_blks + blk;
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
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.col_spillage_start + h_idx * layout.n_blks + blk;
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
    for (h_idx, hydro) in ctx.hydros.iter().enumerate() {
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        let max_div = hydro.diversion.as_ref().map_or(0.0, |d| d.max_flow_m3s);
        for blk in 0..layout.n_blks {
            let col = layout.col_diversion_start + h_idx * layout.n_blks + blk;
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
        let tb = ctx.bounds.thermal_bounds(t_idx, stage_idx);
        let marginal_cost_per_mwh = tb.cost_per_mwh;
        for blk in 0..layout.n_blks {
            let col = layout.col_thermal_start + t_idx * layout.n_blks + blk;
            bufs.col_lower[col] = tb.min_generation_mw;
            bufs.col_upper[col] = tb.max_generation_mw;
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col] = marginal_cost_per_mwh * block_hours;
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
        let lb = ctx.bounds.line_bounds(l_idx, stage_idx);
        let lp = ctx.penalties.line_penalties(l_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let (df, rf) = ctx.resolved_exchange_factors.factors(l_idx, stage_idx, blk);
            let col_fwd = layout.col_line_fwd_start + l_idx * layout.n_blks + blk;
            let col_rev = layout.col_line_rev_start + l_idx * layout.n_blks + blk;
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
        let bp = ctx.penalties.bus_penalties(b_idx, stage_idx);
        for (seg_idx, segment) in bus.deficit_segments.iter().enumerate() {
            for blk in 0..layout.n_blks {
                let col_def = layout.col_deficit_start
                    + b_idx * layout.max_deficit_segments * layout.n_blks
                    + seg_idx * layout.n_blks
                    + blk;
                let block_hours = stage.blocks[blk].duration_hours;
                bufs.col_upper[col_def] = segment.depth_mw.unwrap_or(f64::INFINITY);
                bufs.objective[col_def] = segment.cost_per_mwh * block_hours;
            }
        }
        for blk in 0..layout.n_blks {
            let col_exc = layout.col_excess_start + b_idx * layout.n_blks + blk;
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
            let col = layout.col_inflow_slack_start + h_idx;
            let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
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
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.col_generation_start + local_idx * layout.n_blks + blk;
            bufs.col_lower[col] = 0.0;
            bufs.col_upper[col] = hb.max_generation_mw;
        }
    }
}

/// Evaporation columns: 3 per evaporation hydro (`Q_ev`, `f_evap_plus`, `f_evap_minus`).
///
/// All three columns are stage-level (not per-block).  `Q_ev` is bounded
/// symmetrically `[-q_max, +q_max]` so a negative value can absorb net
/// rainfall input on the lake surface; `f_evap_plus` and `f_evap_minus` are
/// bounded `[0, +inf)`.  `Q_ev` carries zero objective cost (evaporation
/// flow itself is not penalised).  `f_evap_plus` and `f_evap_minus` carry
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
        let col_q_ev = layout.col_evap_start + local_idx * 3;
        let col_f_plus = layout.col_evap_start + local_idx * 3 + 1;
        let col_f_minus = layout.col_evap_start + local_idx * 3 + 2;
        // Signed flow: negative Q_ev reads as net rainfall input (inflow).
        // Bound: [-q_max, +q_max] where q_max = |k_evap0 + k_evap_v * v_max| * margin.
        match ctx.evaporation_models.model(h_idx) {
            EvaporationModel::Linearized { coefficients, .. } => {
                let coeff = &coefficients[stage_idx];
                let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
                let q_max_abs = (coeff.k_evap0 + coeff.k_evap_v * hb.max_storage_hm3).abs()
                    * Q_EV_SAFETY_MARGIN;
                bufs.col_lower[col_q_ev] = -q_max_abs;
                bufs.col_upper[col_q_ev] = q_max_abs;
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
        // Q_ev (offset 0) keeps objective = 0.0 (already the vec initialisation default).
        // f_evap_plus = under-evaporation (evaporated less than target).
        // f_evap_minus = over-evaporation (evaporated more than target).
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        bufs.objective[col_f_plus] = hp.evaporation_violation_neg_cost * total_stage_hours;
        bufs.objective[col_f_minus] = hp.evaporation_violation_pos_cost * total_stage_hours;
    }
}

/// Withdrawal violation slack columns — neg (under-withdrawal) and pos (over-withdrawal).
///
/// One stage-level column per hydro for each direction.
/// Bounds `[0, +inf)` when `water_withdrawal_m3s > 0`; pinned to `[0, 0]` otherwise.
/// Pinning to zero when there is no scheduled withdrawal ensures the column has
/// no LP effect (it is presolve-eliminated), preserving identical behaviour to
/// the pre-withdrawal implementation for cases where withdrawal is absent.
fn fill_withdrawal_slack_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    total_stage_hours: f64,
    bufs: &mut ColumnBufs<'_>,
) {
    // Neg slacks (under-withdrawal).
    for h_idx in 0..layout.n_h {
        let col = layout.col_withdrawal_neg_start + h_idx;
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        bufs.col_upper[col] = if hb.water_withdrawal_m3s > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        bufs.objective[col] = hp.water_withdrawal_violation_neg_cost * total_stage_hours;
    }
    // Pos slacks (over-withdrawal).
    for h_idx in 0..layout.n_h {
        let col = layout.col_withdrawal_pos_start + h_idx;
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        bufs.col_upper[col] = if hb.water_withdrawal_m3s > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        bufs.objective[col] = hp.water_withdrawal_violation_pos_cost * total_stage_hours;
    }
}

/// Operational violation slack columns: 4 families of `n_h * n_blks` columns.
///
/// Delegates to one sub-helper per family; each sub-helper is independent
/// and writes to a disjoint column range.
fn fill_operational_slack_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    fill_outflow_below_columns(ctx, stage, stage_idx, layout, bufs);
    fill_outflow_above_columns(ctx, stage, stage_idx, layout, bufs);
    fill_turbine_below_columns(ctx, stage, stage_idx, layout, bufs);
    fill_generation_below_columns(ctx, stage, stage_idx, layout, bufs);
}

/// Outflow-below-minimum slack columns (`sigma_outflow_below_{h,k}`).
///
/// Active (unbounded above) when `min_outflow_m3s > 0`; pinned to `[0, 0]` otherwise.
fn fill_outflow_below_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.col_outflow_below_start + h_idx * layout.n_blks + blk;
            bufs.col_upper[col] = if hb.min_outflow_m3s > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
            bufs.objective[col] =
                hp.outflow_violation_below_cost * stage.blocks[blk].duration_hours;
        }
    }
}

/// Outflow-above-maximum slack columns (`sigma_outflow_above_{h,k}`).
///
/// Active (unbounded above) when `max_outflow_m3s` is `Some`; pinned to `[0, 0]` otherwise.
fn fill_outflow_above_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.col_outflow_above_start + h_idx * layout.n_blks + blk;
            bufs.col_upper[col] = if hb.max_outflow_m3s.is_some() {
                f64::INFINITY
            } else {
                0.0
            };
            bufs.objective[col] =
                hp.outflow_violation_above_cost * stage.blocks[blk].duration_hours;
        }
    }
}

/// Turbine-below-minimum slack columns (`sigma_turbine_below_{h,k}`).
///
/// Active (unbounded above) when `min_turbined_m3s > 0`; pinned to `[0, 0]` otherwise.
fn fill_turbine_below_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.col_turbine_below_start + h_idx * layout.n_blks + blk;
            bufs.col_upper[col] = if hb.min_turbined_m3s > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
            bufs.objective[col] =
                hp.turbined_violation_below_cost * stage.blocks[blk].duration_hours;
        }
    }
}

/// Generation-below-minimum slack columns (`sigma_generation_below_{h,k}`).
///
/// Active (unbounded above) when `min_generation_mw > 0`; pinned to `[0, 0]` otherwise.
fn fill_generation_below_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.col_generation_below_start + h_idx * layout.n_blks + blk;
            bufs.col_upper[col] = if hb.min_generation_mw > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
            bufs.objective[col] =
                hp.generation_violation_below_cost * stage.blocks[blk].duration_hours;
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
/// level on every stage (must-run, matching NEWAVE's
/// `geracao_usinas_nao_simuladas` pre-netting convention).
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
            .resolved_ncs_bounds
            .available_generation(ncs_sys_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.col_ncs_start + ncs_local * layout.n_blks + blk;
            let factor = ctx.resolved_ncs_factors.factor(ncs_sys_idx, stage_idx, blk);
            let upper = avail_gen * factor;
            bufs.col_upper[col] = upper;
            bufs.col_lower[col] = if ncs.allow_curtailment { 0.0 } else { upper };
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col] = -ncs.curtailment_cost * block_hours;
        }
    }
}

/// Z-inflow columns: free variables for realized total inflow per hydro.
///
/// `col_lower = -inf`, `col_upper = +inf`, `objective = 0.0` (no direct cost).
fn fill_z_inflow_columns(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    for h_idx in 0..layout.n_h {
        let col = layout.col_z_inflow_start + h_idx;
        bufs.col_lower[col] = f64::NEG_INFINITY;
        bufs.col_upper[col] = f64::INFINITY;
        // objective[col] = 0.0 — already zero from vec initialisation.
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

    // Water balance rows: static RHS = ζ * (deterministic_base_h - water_withdrawal_m3s_h).
    // The withdrawal is a fixed schedule that reduces the effective inflow available
    // to the reservoir. Subtracting it from the base keeps the row bound correct for
    // the stage template; the PAR(p) noise innovation is added at solve time via patches.
    for h_idx in 0..layout.n_h {
        let row = layout.row_water_balance_start + h_idx;
        let base = if ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == layout.n_h {
            ctx.par_lp.deterministic_base(stage_idx, h_idx)
        } else {
            0.0
        };
        let withdrawal = ctx
            .bounds
            .hydro_bounds(h_idx, stage_idx)
            .water_withdrawal_m3s;
        let rhs = layout.zeta * (base - withdrawal);
        row_lower[row] = rhs;
        row_upper[row] = rhs;
    }

    // Load balance rows: static RHS = mean_mw * block_factor.
    // Block factors from `load_factors.json` scale the mean load per block
    // (e.g., heavy/medium/light blocks). Default factor is 1.0 (no scaling).
    for (b_idx, bus) in ctx.buses.iter().enumerate() {
        let mean_mw = ctx
            .load_models
            .iter()
            .find(|lm| lm.bus_id == bus.id && lm.stage_id == stage.id)
            .map_or(0.0, |lm| lm.mean_mw);
        for blk in 0..layout.n_blks {
            let factor = ctx.resolved_load_factors.factor(b_idx, stage_idx, blk);
            let row = layout.row_load_balance_start + b_idx * layout.n_blks + blk;
            let rhs = mean_mw * factor;
            row_lower[row] = rhs;
            row_upper[row] = rhs;
        }
    }

    // FPHA constraint rows: g_{h,k} - gamma_v/2*v - gamma_v/2*v_in
    //                            - gamma_q*q - gamma_s*s <= gamma_0
    // Row lower = -INF, row upper = gamma_0 (pre-scaled intercept).
    // The v_in contribution is encoded in the matrix entry on the v_in column,
    // so the static upper bound only needs the intercept term.
    let n_blks = layout.n_blks;
    for (local_idx, &h_idx) in layout.fpha_hydro_indices.iter().enumerate() {
        if let ResolvedProductionModel::Fpha { planes, .. } =
            ctx.production_models.model(h_idx, stage_idx)
        {
            let n_planes = planes.len();
            debug_assert_eq!(
                n_planes, layout.fpha_planes_per_hydro[local_idx],
                "plane count mismatch for FPHA hydro {h_idx} at stage {stage_idx}"
            );
            for blk in 0..n_blks {
                for (p_idx, plane) in planes.iter().enumerate() {
                    let row = layout.row_fpha_start
                        + local_idx * n_blks * n_planes
                        + blk * n_planes
                        + p_idx;
                    row_lower[row] = f64::NEG_INFINITY;
                    row_upper[row] = plane.intercept;
                }
            }
        }
    }

    // Evaporation constraint rows: Q_ev = k_evap0 + k_evap_v/2*(v + v_in - 2*V_ref).
    // The volume-dependent term `k_evap_v/2 * v` is added via the CSC matrix entry
    // on the outgoing-storage column, so the static row bounds only encode the
    // constant term `k_evap0`.  The constraint is an equality:
    // row_lower == row_upper == k_evap0.
    for (local_idx, &h_idx) in layout.evap_hydro_indices.iter().enumerate() {
        if let EvaporationModel::Linearized { coefficients, .. } =
            ctx.evaporation_models.model(h_idx)
        {
            debug_assert!(
                stage_idx < coefficients.len(),
                "stage index {stage_idx} out of bounds for evaporation coefficients (len = {})",
                coefficients.len()
            );
            let k_evap0 = coefficients[stage_idx].k_evap0;
            let row = layout.row_evap_start + local_idx;
            row_lower[row] = k_evap0;
            row_upper[row] = k_evap0;
        }
    }

    // Operational violation constraint rows (min/max outflow, min turbine, min generation).
    fill_operational_violation_rows(
        ctx,
        stage,
        stage_idx,
        layout,
        &mut row_lower,
        &mut row_upper,
    );

    // Z-inflow definition rows: equality constraints with RHS = base_h (m3/s).
    // The base is the deterministic PAR base inflow (before noise), NOT multiplied
    // by zeta and NOT reduced by withdrawal. The noise component (sigma * eta) is
    // added at solve time via PatchBuffer Category 5.
    for h_idx in 0..layout.n_h {
        let row = layout.row_z_inflow_start + h_idx;
        let base = if ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == layout.n_h {
            ctx.par_lp.deterministic_base(stage_idx, h_idx)
        } else {
            0.0
        };
        row_lower[row] = base;
        row_upper[row] = base;
    }

    (row_lower, row_upper)
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
    // Min outflow rows (>= constraint): LHS + sigma >= min_outflow_m3s
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let row = layout.row_min_outflow_start + h_idx * layout.n_blks + blk;
            row_lower[row] = hb.min_outflow_m3s;
            row_upper[row] = f64::INFINITY;
        }
    }

    // Max outflow rows (<= constraint): LHS - sigma <= max_outflow_m3s
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let row = layout.row_max_outflow_start + h_idx * layout.n_blks + blk;
            row_lower[row] = f64::NEG_INFINITY;
            row_upper[row] = hb.max_outflow_m3s.unwrap_or(f64::INFINITY);
        }
    }

    // Min turbine flow rows (>= constraint): LHS + sigma >= min_turbined_m3s
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let row = layout.row_min_turbine_start + h_idx * layout.n_blks + blk;
            row_lower[row] = hb.min_turbined_m3s;
            row_upper[row] = f64::INFINITY;
        }
    }

    // Min generation rows (>= constraint): LHS + sigma >= min_generation_mw
    for h_idx in 0..layout.n_h {
        let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let row = layout.row_min_generation_start + h_idx * layout.n_blks + blk;
            row_lower[row] = hb.min_generation_mw;
            row_upper[row] = f64::INFINITY;
        }
    }
}

/// Build the CSC matrix entries for one stage.
///
/// Returns one `Vec<(row, value)>` per column. Entries within each column are
/// sorted by row index before return (CSC invariant).
/// Fill state-region and water-balance entries into `col_entries`.
///
/// Writes entries for storage-fixing rows, AR lag-fixing rows,
/// and the water-balance rows (outgoing/incoming storage, turbine, spillage,
/// upstream cascade, and AR lag dynamics).
pub(super) fn fill_state_and_water_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let idx = StageIndexer::new(ctx.n_hydros, ctx.max_par_order);
    let n_h = layout.n_h;
    let n_blks = layout.n_blks;
    let lag_order = layout.lag_order;
    let zeta = layout.zeta;
    let row_water = layout.row_water_balance_start;

    // State rows: storage-fixing (incoming storage column → row h).
    for h in 0..n_h {
        let col = idx.storage_in.start + h;
        col_entries[col].push((h, 1.0));
    }

    // State rows: lag-fixing (lag column → diagonal row).
    for lag in 0..lag_order {
        for h in 0..n_h {
            let col = idx.inflow_lags.start + lag * n_h + h;
            let row = n_h + lag * n_h + h;
            col_entries[col].push((row, 1.0));
        }
    }

    // Water balance: outgoing storage (+1), incoming storage (-1),
    // turbine/spillage (+tau), upstream turbine/spillage (-tau),
    // and AR lag dynamics (-ζ*ψ).
    for h_idx in 0..n_h {
        let hydro = &ctx.hydros[h_idx];
        let row = row_water + h_idx;
        col_entries[h_idx].push((row, 1.0));
        col_entries[idx.storage_in.start + h_idx].push((row, -1.0));
        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col_turbine = layout.col_turbine_start + h_idx * n_blks + blk;
            col_entries[col_turbine].push((row, tau_h));
            let col_spillage = layout.col_spillage_start + h_idx * n_blks + blk;
            col_entries[col_spillage].push((row, tau_h));
            // Diversion outflow: this hydro's diversion column enters its own
            // water balance with +tau (outflow), same sign as turbine/spillage.
            let col_diversion = layout.col_diversion_start + h_idx * n_blks + blk;
            col_entries[col_diversion].push((row, tau_h));
            // Cascade inflow: upstream turbine/spillage enter with -tau.
            for &up_id in ctx.cascade.upstream(hydro.id) {
                if let Some(&u_idx) = ctx.hydro_pos.get(&up_id) {
                    col_entries[layout.col_turbine_start + u_idx * n_blks + blk]
                        .push((row, -tau_h));
                    col_entries[layout.col_spillage_start + u_idx * n_blks + blk]
                        .push((row, -tau_h));
                }
            }
            // Diversion inflow: for each hydro that diverts TO this hydro, its
            // diversion column enters this hydro's water balance with -tau.
            if let Some(sources) = ctx.diversion_upstream.get(&hydro.id) {
                for &d_idx in sources {
                    let col_div = layout.col_diversion_start + d_idx * n_blks + blk;
                    col_entries[col_div].push((row, -tau_h));
                }
            }
        }
        if ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == n_h {
            let psi = ctx.par_lp.psi_slice(stage_idx, h_idx);
            for (lag, &psi_val) in psi.iter().enumerate() {
                if psi_val != 0.0 && lag < lag_order {
                    let col = idx.inflow_lags.start + lag * n_h + h_idx;
                    col_entries[col].push((row, -zeta * psi_val));
                }
            }
        }
    }

    // Inflow non-negativity slack: sigma_inf_h enters water balance with -ζ.
    if ctx.has_penalty {
        for h_idx in 0..n_h {
            let col = layout.col_inflow_slack_start + h_idx;
            let row = row_water + h_idx;
            col_entries[col].push((row, -zeta));
        }
    }

    // Evaporation: Q_ev_h enters water balance with +ζ.
    // Evaporation is an outflow (water leaving the reservoir), so its
    // coefficient matches the turbine/spillage sign convention (positive).
    for (local_idx, &h_idx) in layout.evap_hydro_indices.iter().enumerate() {
        let col_q_ev = layout.col_evap_start + local_idx * 3;
        let row = row_water + h_idx;
        col_entries[col_q_ev].push((row, zeta));
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

/// Fill load-balance entries into `col_entries`.
///
/// Writes entries for hydro turbine generation, thermal generation,
/// line forward/reverse flows, and deficit/excess slacks.
///
/// For FPHA hydros the generation variable `g_{h,k}` (in `col_generation_start`)
/// enters the load balance with coefficient +1.0 instead of `rho * turbine_col`.
/// For constant-productivity hydros the original `rho * turbine_col` behavior is unchanged.
pub(super) fn fill_load_balance_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let row_load = layout.row_load_balance_start;

    // Build a quick lookup from hydro index to FPHA local index.
    // `fpha_local[h_idx]` is `Some(local_idx)` if hydro `h_idx` uses FPHA at this stage.
    let mut fpha_local: Vec<Option<usize>> = vec![None; ctx.n_hydros];
    for (local_idx, &h_idx) in layout.fpha_hydro_indices.iter().enumerate() {
        fpha_local[h_idx] = Some(local_idx);
    }

    // Hydro contribution to load balance.
    // - FPHA hydros: g_{h,k} column with coefficient +1.0
    // - Constant-productivity hydros: rho * turbine_col (unchanged)
    for (h_idx, hydro) in ctx.hydros.iter().enumerate() {
        if let Some(local_idx) = fpha_local[h_idx] {
            // FPHA: use the generation variable column.
            // The resolved model for this stage must be Fpha.
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
                    let col = layout.col_generation_start + local_idx * n_blks + blk;
                    col_entries[col].push((row, 1.0));
                }
            }
        } else {
            // Constant productivity: use the resolved per-stage production model
            // from hydro_production_models.json.
            let rho = match ctx.production_models.model(h_idx, stage_idx) {
                ResolvedProductionModel::ConstantProductivity { productivity } => *productivity,
                ResolvedProductionModel::Fpha { .. } => {
                    // This branch should not be reached because FPHA hydros are handled
                    // by the `if let Some(local_idx) = fpha_local[h_idx]` branch above.
                    // If we get here, the FPHA local-index table is inconsistent.
                    unreachable!(
                        "non-FPHA branch reached for FPHA resolved model at hydro {h_idx}"
                    );
                }
            };
            if let Some(&b_idx) = ctx.bus_pos.get(&hydro.bus_id) {
                for blk in 0..n_blks {
                    let row = row_load + b_idx * n_blks + blk;
                    let col = layout.col_turbine_start + h_idx * n_blks + blk;
                    col_entries[col].push((row, rho));
                }
            }
        }
    }

    // Thermal generation.
    for (t_idx, thermal) in ctx.thermals.iter().enumerate() {
        if let Some(&b_idx) = ctx.bus_pos.get(&thermal.bus_id) {
            for blk in 0..n_blks {
                let row = row_load + b_idx * n_blks + blk;
                let col = layout.col_thermal_start + t_idx * n_blks + blk;
                col_entries[col].push((row, 1.0));
            }
        }
    }

    // Line flows (+1 at target, -1 at source for forward; reversed for reverse).
    for (l_idx, line) in ctx.lines.iter().enumerate() {
        let src_idx = ctx.bus_pos.get(&line.source_bus_id).copied();
        let tgt_idx = ctx.bus_pos.get(&line.target_bus_id).copied();
        for blk in 0..n_blks {
            let col_fwd = layout.col_line_fwd_start + l_idx * n_blks + blk;
            let col_rev = layout.col_line_rev_start + l_idx * n_blks + blk;
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

    // Deficit (+1 for every segment) and excess (-1).
    //
    // All deficit segment columns for bus `b_idx` at block `blk` enter the same
    // load-balance row with coefficient +1.0 (total deficit = sum of segments).
    for (b_idx, bus) in ctx.buses.iter().enumerate() {
        for blk in 0..n_blks {
            let row = row_load + b_idx * n_blks + blk;
            for seg_idx in 0..bus.deficit_segments.len() {
                let col_def = layout.col_deficit_start
                    + b_idx * layout.max_deficit_segments * n_blks
                    + seg_idx * n_blks
                    + blk;
                col_entries[col_def].push((row, 1.0));
            }
            let col_exc = layout.col_excess_start + b_idx * n_blks + blk;
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
pub(super) fn fill_fpha_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let idx = StageIndexer::new(ctx.n_hydros, ctx.max_par_order);
    let n_blks = layout.n_blks;

    for (local_idx, &h_idx) in layout.fpha_hydro_indices.iter().enumerate() {
        let model = ctx.production_models.model(h_idx, stage_idx);
        let planes = match model {
            ResolvedProductionModel::Fpha { planes, .. } => planes,
            ResolvedProductionModel::ConstantProductivity { .. } => {
                // Should never happen: fpha_hydro_indices only contains FPHA hydros.
                debug_assert!(
                    false,
                    "fpha_hydro_indices contains hydro {h_idx} but model is ConstantProductivity"
                );
                continue;
            }
        };
        let n_planes = planes.len();

        for blk in 0..n_blks {
            // Column indices for this hydro/block.
            let col_v = h_idx; // outgoing storage column
            let col_v_in = idx.storage_in.start + h_idx; // incoming storage column
            let col_q = layout.col_turbine_start + h_idx * n_blks + blk;
            let col_s = layout.col_spillage_start + h_idx * n_blks + blk;
            let col_g = layout.col_generation_start + local_idx * n_blks + blk;

            for (p_idx, plane) in planes.iter().enumerate() {
                let row =
                    layout.row_fpha_start + local_idx * n_blks * n_planes + blk * n_planes + p_idx;

                // g_{h,k} column: +1.0
                col_entries[col_g].push((row, 1.0));
                // v (outgoing storage): -gamma_v/2
                col_entries[col_v].push((row, -plane.gamma_v / 2.0));
                // v_in (incoming storage, fixed by storage-fixing row): -gamma_v/2
                col_entries[col_v_in].push((row, -plane.gamma_v / 2.0));
                // q_{h,k} (turbine): -gamma_q
                col_entries[col_q].push((row, -plane.gamma_q));
                // s_{h,k} (spillage): -gamma_s
                col_entries[col_s].push((row, -plane.gamma_s));
            }
        }
    }
}

/// Fill CSC matrix entries for the evaporation constraint rows.
///
/// For each evaporation hydro `h` at local position `local_idx`, the equality row
/// `row_evap_start + local_idx` encodes:
///
/// ```text
/// Q_ev_h  column:  +1.0
/// v_h     column:  -k_evap_v / 2      (outgoing storage)
/// v_in_h  column:  -k_evap_v / 2      (incoming storage; fixed by storage-fixing row)
/// f_plus  column:  +1.0
/// f_minus column:  -1.0
/// ```
///
/// These entries implement `Q_ev - k_evap_v/2*v - k_evap_v/2*v_in + f_plus - f_minus = k_evap0`,
/// where `k_evap0` is already encoded in the row bounds set by `fill_stage_rows`.
/// When `v_in` is fixed to value `V`, the effective RHS becomes `k_evap0 + k_evap_v/2 * V`,
/// which matches the linearized evaporation at the average volume `(v + V) / 2`.
pub(super) fn fill_evaporation_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let idx = StageIndexer::new(ctx.n_hydros, ctx.max_par_order);

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

        let col_q_ev = layout.col_evap_start + local_idx * 3;
        let col_f_plus = layout.col_evap_start + local_idx * 3 + 1;
        let col_f_minus = layout.col_evap_start + local_idx * 3 + 2;
        let col_v = h_idx; // outgoing storage column
        let col_v_in = idx.storage_in.start + h_idx; // incoming storage column

        let row = layout.row_evap_start + local_idx;

        // Q_ev_h: +1.0
        col_entries[col_q_ev].push((row, 1.0));
        // v_h (outgoing storage): -k_evap_v / 2
        col_entries[col_v].push((row, -coeff.k_evap_v / 2.0));
        // v_in_h (incoming storage, fixed by storage-fixing row): -k_evap_v / 2
        col_entries[col_v_in].push((row, -coeff.k_evap_v / 2.0));
        // f_evap_plus: +1.0
        col_entries[col_f_plus].push((row, 1.0));
        // f_evap_minus: -1.0
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
    /// Column lower bounds (currently unused for generic constraints).
    pub(super) _col_lower: &'a mut [f64],
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

    // Build a full StageIndexer so that resolve_variable_ref can map any
    // VariableRef to the correct column index for this stage.
    let indexer = crate::indexer::StageIndexer::with_equipment_and_evaporation(
        &crate::indexer::EquipmentCounts {
            hydro_count: ctx.n_hydros,
            max_par_order: ctx.max_par_order,
            n_thermals: ctx.n_thermals,
            n_lines: ctx.n_lines,
            n_buses: ctx.n_buses,
            n_blks: layout.n_blks,
            has_inflow_penalty: ctx.has_penalty,
            max_deficit_segments: layout.max_deficit_segments,
        },
        &crate::indexer::FphaColumnLayout {
            hydro_indices: layout.fpha_hydro_indices.clone(),
            planes_per_hydro: layout.fpha_planes_per_hydro.clone(),
        },
        &crate::indexer::EvapConfig {
            hydro_indices: layout.evap_hydro_indices.clone(),
        },
    );

    let positions = crate::generic_constraints::EntityPositionMaps {
        hydro: &ctx.hydro_pos,
        thermal: &ctx.thermal_pos,
        bus: &ctx.bus_pos,
        line: &ctx.line_pos,
    };

    for (entry_idx, entry) in layout.generic_constraint_rows.iter().enumerate() {
        let row = layout.row_generic_start + entry_idx;
        let constraint = &ctx.generic_constraints[entry.constraint_idx];
        let block_hours = stage.blocks[entry.block_idx].duration_hours;

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
                &indexer,
                ctx.production_models,
                &positions,
            );
            for (col, multiplier) in pairs {
                let coef = match term.coefficient {
                    CoefficientRef::Literal(v) => v,
                    CoefficientRef::Parameter(param_id) => {
                        ctx.resolved_parameters.get(param_id, stage_idx)
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
            let row = layout.row_load_balance_start + bus_idx * layout.n_blks + blk;
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
    let idx = StageIndexer::new(n_h, lag_order);

    for h_idx in 0..n_h {
        let row = layout.row_z_inflow_start + h_idx;

        // z_h column: coefficient +1.0
        let col_z = layout.col_z_inflow_start + h_idx;
        col_entries[col_z].push((row, 1.0));

        // Lag columns: coefficient -psi_l for each nonzero psi.
        // Uses lag-major layout (lag * n_h + h) matching the water-balance
        // AR dynamics entries in fill_state_and_water_entries.
        if ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == n_h {
            let psi = ctx.par_lp.psi_slice(stage_idx, h_idx);
            for (lag, &psi_val) in psi.iter().enumerate() {
                if psi_val != 0.0 && lag < lag_order {
                    let col = idx.inflow_lags.start + lag * n_h + h_idx;
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

    // Build FPHA local-index lookup (same pattern as fill_load_balance_entries).
    let mut fpha_local: Vec<Option<usize>> = vec![None; ctx.n_hydros];
    for (local_idx, &h_idx) in layout.fpha_hydro_indices.iter().enumerate() {
        fpha_local[h_idx] = Some(local_idx);
    }

    for (h_idx, fpha_local_entry) in fpha_local.iter().enumerate() {
        // ── Min outflow (per block): q + s + d + sigma >= min_outflow_m3s ───
        for blk in 0..n_blks {
            let row = layout.row_min_outflow_start + h_idx * n_blks + blk;
            let col_q = layout.col_turbine_start + h_idx * n_blks + blk;
            col_entries[col_q].push((row, 1.0));
            let col_s = layout.col_spillage_start + h_idx * n_blks + blk;
            col_entries[col_s].push((row, 1.0));
            let col_d = layout.col_diversion_start + h_idx * n_blks + blk;
            col_entries[col_d].push((row, 1.0));
            let col_slack = layout.col_outflow_below_start + h_idx * n_blks + blk;
            col_entries[col_slack].push((row, 1.0));
        }

        // ── Max outflow (per block): q + s + d - sigma <= max_outflow_m3s ───
        for blk in 0..n_blks {
            let row = layout.row_max_outflow_start + h_idx * n_blks + blk;
            let col_q = layout.col_turbine_start + h_idx * n_blks + blk;
            col_entries[col_q].push((row, 1.0));
            let col_s = layout.col_spillage_start + h_idx * n_blks + blk;
            col_entries[col_s].push((row, 1.0));
            let col_d = layout.col_diversion_start + h_idx * n_blks + blk;
            col_entries[col_d].push((row, 1.0));
            let col_slack = layout.col_outflow_above_start + h_idx * n_blks + blk;
            col_entries[col_slack].push((row, -1.0));
        }

        // ── Min turbine flow (per block): q + sigma >= min_turbined_m3s ─────
        for blk in 0..n_blks {
            let row = layout.row_min_turbine_start + h_idx * n_blks + blk;
            let col_q = layout.col_turbine_start + h_idx * n_blks + blk;
            col_entries[col_q].push((row, 1.0));
            let col_slack = layout.col_turbine_below_start + h_idx * n_blks + blk;
            col_entries[col_slack].push((row, 1.0));
        }

        // ── Min generation (per block): g + sigma >= min_generation_mw ──────
        if let Some(&local_fpha_idx) = fpha_local_entry.as_ref() {
            // FPHA: generation variable g_{h,blk} (already in MW).
            for blk in 0..n_blks {
                let row = layout.row_min_generation_start + h_idx * n_blks + blk;
                let col_g = layout.col_generation_start + local_fpha_idx * n_blks + blk;
                col_entries[col_g].push((row, 1.0));
                let col_slack = layout.col_generation_below_start + h_idx * n_blks + blk;
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
                let col_q = layout.col_turbine_start + h_idx * n_blks + blk;
                col_entries[col_q].push((row, rho));
                let col_slack = layout.col_generation_below_start + h_idx * n_blks + blk;
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
    fill_load_balance_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_ncs_load_balance_entries(ctx, layout, &mut col_entries);
    fill_fpha_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_evaporation_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_z_inflow_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_operational_violation_entries(ctx, stage, stage_idx, layout, &mut col_entries);

    col_entries
}

/// Assemble CSC arrays from per-column entry lists.
///
/// Returns `(col_starts, row_indices, values)` in the format required by
/// `SolverInterface::load_model`.
pub(super) fn assemble_csc(col_entries: &[Vec<(usize, f64)>]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
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

    use crate::energy_conversion::{build_hydro_energy_productivity_override, EnergyConversionSet};
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
            gnl_config: None,
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
            build_hydro_energy_productivity_override(vec![]).expect("empty override table");
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
            build_hydro_energy_productivity_override(vec![]).expect("empty override table");
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
            build_hydro_energy_productivity_override(vec![]).expect("empty override table");
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
