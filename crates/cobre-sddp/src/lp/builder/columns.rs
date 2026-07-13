use cobre_core::commissioning::{Phase, commissioning_active, filling_phase};
use cobre_core::{ContractType, Stage};

use crate::hydro_models::{EvaporationModel, ResolvedProductionModel};
use crate::indexer::{
    AnticipatedLocal, BlockIdx, Boundary, EvapLocal, FillingTargetLocal, FloorLocal, FphaLocal,
    HydroSys, LineSys,
};

use super::EVAPORATION_FLOW_SAFETY_MARGIN;
use super::layout::{StageLayout, TemplateBuildCtx};
use crate::generic_constraints::contract_family_slot;

/// Mutable column-bound and objective buffers shared by all fill helpers.
pub(super) struct ColumnBufs<'a> {
    /// Column lower bounds.
    pub(super) col_lower: &'a mut [f64],
    /// Column upper bounds.
    pub(super) col_upper: &'a mut [f64],
    /// Objective coefficients.
    pub(super) objective: &'a mut [f64],
}

/// Fill column lower/upper bounds and objective coefficients for one stage.
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

    fill_storage_columns(ctx, stage, stage_idx, layout, b);
    fill_transit_bucket_columns(layout, b);
    fill_anticipated_slot_columns(layout, b);
    fill_ar_lag_columns(layout, b);
    fill_anticipated_state_columns(layout, b);
    fill_theta_column(layout, b);
    fill_turbine_columns(ctx, stage, stage_idx, layout, b);
    fill_spillage_columns(ctx, stage, stage_idx, layout, b);
    fill_diversion_columns(ctx, stage, stage_idx, layout, b);
    fill_thermal_columns(ctx, stage, stage_idx, layout, b);
    fill_anticipated_columns(ctx, stage_idx, layout, b);
    fill_line_columns(ctx, stage, stage_idx, layout, b);
    fill_deficit_and_excess_columns(ctx, stage, stage_idx, layout, b);
    fill_inflow_slack_columns(ctx, stage_idx, layout, total_stage_hours, b);
    fill_fpha_generation_columns(ctx, stage_idx, layout, b);
    fill_evaporation_columns(ctx, stage, stage_idx, layout, b);
    fill_withdrawal_slack_columns(ctx, stage_idx, layout, total_stage_hours, b);
    fill_operational_slack_columns(ctx, stage, stage_idx, layout, b);
    fill_ncs_columns(ctx, stage, stage_idx, layout, b);
    fill_pumping_columns(ctx, stage, stage_idx, layout, b);
    fill_contract_columns(ctx, stage, stage_idx, layout, b);
    fill_filling_target_columns(ctx, stage_idx, layout, b);
    fill_filled_min_storage_floor_columns(ctx, stage_idx, layout, b);
    fill_z_inflow_columns(layout, b);

    (col_lower, col_upper, objective)
}

/// Outgoing and incoming storage columns.
///
/// Incoming storage `v_in_h` is left unconstrained here — pinned at solve time via
/// column bounds, not this site.
fn fill_storage_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let hydro = &ctx.hydros[h_idx];
        // CONTRACT: `min_storage` is a HARD lower bound for every hydro EXCEPT (a) a
        // filling one (`hydro.filling.is_some()`), whose floor relaxes to `0` in ALL
        // phases and is re-imposed as a SOFT floor by
        // `fill_filled_min_storage_floor_columns`, and (b) a non-filling hydro while
        // `PreFilling` (commissioning-dormant), whose frozen-identity row pins `v_h`
        // to the inert IC storage — a hard floor above that IC value would reject the
        // pin and make the LP infeasible. FORBIDDEN — globalizing the relax to all
        // Operating hydros (makes dead volume soft system-wide); keeping it hard
        // through a dormant non-filling stage (rejects the IC pin). The dormant relax
        // disappears at `Operating`, restoring the hard floor.
        let floor_off = hydro.filling.is_some()
            || matches!(
                filling_phase(
                    hydro.filling.as_ref(),
                    hydro.entry_stage_id,
                    hydro.exit_stage_id,
                    stage.id,
                ),
                Phase::PreFilling
            );
        let hb = ctx.resolved.bounds.hydro_bounds(h_idx, stage_idx);
        let storage_lower = if floor_off { 0.0 } else { hb.min_storage_hm3 };
        bufs.col_lower[h_idx] = storage_lower;
        bufs.col_upper[h_idx] = hb.max_storage_hm3;
        bufs.col_lower[layout.col_storage_in_start() + h_idx] = f64::NEG_INFINITY;
        bufs.col_upper[layout.col_storage_in_start() + h_idx] = f64::INFINITY;
        // Interior Sᵏ reuse the outgoing column's EXACT bounds, floor_off included: the
        // frozen-identity chain pins each interior to the inert IC, so a hard floor above
        // IC would reject the pin.
        for k in 1..layout.n_blks {
            let col = layout.block_storage_col(HydroSys::new(h_idx), Boundary::Interior(k));
            bufs.col_lower[col] = storage_lower;
            bufs.col_upper[col] = hb.max_storage_hm3;
        }
    }
}

/// Travel-time bucket state columns.
///
/// A masked lag (no definition row, `entries::fill_transit_bucket_definition_entries`)
/// gets its outgoing column frozen `[0, 0]` here — the column-freeze half of the
/// two-sided masking contract; leaving it free would be a free column with no defining
/// constraint. Incoming buckets stay open, pinned every solve by `fill_col_state_patches`.
fn fill_transit_bucket_columns(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    let state = layout.state;
    for range in super::entries::transit_bucket_plant_ranges(state) {
        let col_base = state.transit_buckets_out.start + range.start;
        let ring = super::entries::transit_bucket_ring(state, range.clone());
        ring.freeze_masked_columns(
            &layout.rows.transit_bucket_row_pos[range],
            col_base,
            (0.0, f64::INFINITY),
            bufs,
        );
    }
}

/// Anticipated-ring outgoing (interior + padding) columns: open `(-inf, inf)` bounds
/// (a committed MW value carries either sign, unlike the water buckets' `[0, inf)`) for
/// every reachable slot, frozen `[0, 0]` otherwise (the masked column-freeze, mirroring
/// [`fill_transit_bucket_columns`]). A plant's own newest slot is bounded later by
/// [`fill_anticipated_columns`], which overwrites this fill.
fn fill_anticipated_slot_columns(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    let base = layout.anticipated.col_anticipated_slots_out_start;
    let ring = super::entries::anticipated_ring(layout);
    ring.freeze_masked_columns(
        &layout.anticipated.anticipated_slot_row_pos,
        base,
        (f64::NEG_INFINITY, f64::INFINITY),
        bufs,
    );
}

/// AR lag columns: unconstrained (signed).
fn fill_ar_lag_columns(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    let n_lag_cols = layout.lag_order * layout.n_h;
    for lag_col in layout.col_inflow_lags_start()..layout.col_inflow_lags_start() + n_lag_cols {
        bufs.col_lower[lag_col] = f64::NEG_INFINITY;
        bufs.col_upper[lag_col] = f64::INFINITY;
    }
}

/// Incoming anticipated-ring columns: open `(-INF, +INF)`, left open because pinning
/// is via `set_col_bounds` at solve time (`fill_col_state_patches`), not an equality row.
fn fill_anticipated_state_columns(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    for slot in 0..layout.k_max {
        for plant in 0..layout.n_anticipated {
            let col = layout.col_anticipated_state_start() + slot * layout.n_anticipated + plant;
            bufs.col_lower[col] = f64::NEG_INFINITY;
            bufs.col_upper[col] = f64::INFINITY;
        }
    }
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
/// A suspended hydro (`PreFilling`/`Filling`) forces BOTH bounds to `[0, 0]`: both
/// must drop, or a positive `min_turbined_m3s` leaves the infeasible `[min > 0, 0]`.
fn fill_turbine_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let hydro = &ctx.hydros[h_idx];
        let suspended = matches!(
            filling_phase(
                hydro.filling.as_ref(),
                hydro.entry_stage_id,
                hydro.exit_stage_id,
                stage.id,
            ),
            Phase::PreFilling | Phase::Filling
        );
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
            let col = layout.turbine_col(HydroSys::new(h_idx), BlockIdx::new(blk));
            bufs.col_lower[col] = 0.0;
            bufs.col_upper[col] = if suspended { 0.0 } else { turb_upper };
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col] = hp.turbined_cost * block_hours;
        }
    }
}

/// Spillage columns per hydro per block.
///
/// CONTRACT: a `PreFilling` hydro's spillage is pinned `[0, 0]` (no dam exists yet
/// to spill from), gated on `Phase::PreFilling` ALONE. Two forbidden alternatives:
/// extending the freeze to `Filling` kills the legitimate over-dam relief valve an
/// impounding reservoir needs (D40); gating on `filling.is_none()` leaves the
/// phantom-spill hole open for a filling hydro in its own `PreFilling` sub-phase
/// (D38/D39), where a free spillage column decoupled from frozen storage injects
/// water it does not have onto the downstream balance row.
fn fill_spillage_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let hydro = &ctx.hydros[h_idx];
        let prefilling = matches!(
            filling_phase(
                hydro.filling.as_ref(),
                hydro.entry_stage_id,
                hydro.exit_stage_id,
                stage.id,
            ),
            Phase::PreFilling
        );
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.spillage_col(HydroSys::new(h_idx), BlockIdx::new(blk));
            bufs.col_upper[col] = if prefilling { 0.0 } else { f64::INFINITY };
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col] = hp.spillage_cost * block_hours;
        }
    }
}

/// Diversion columns per hydro per block (dense; non-diverting hydros get `[0, 0]`
/// and are presolve-eliminated).
///
/// A filling hydro (`hydro.filling.is_some()`) is forced to `[0, 0]` in ALL phases —
/// gated on `is_some()`, NOT the `Phase`: phase-gating would wrongly re-enable diversion
/// at `Operating`, but a filling hydro never diverts.
fn fill_diversion_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (h_idx, hydro) in ctx.hydros.iter().enumerate() {
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        let suspended = matches!(
            filling_phase(
                hydro.filling.as_ref(),
                hydro.entry_stage_id,
                hydro.exit_stage_id,
                stage.id,
            ),
            Phase::PreFilling | Phase::Filling
        );
        // CONTRACT: read the per-stage RESOLVED `max_diversion_m3s`, NOT the
        // declaration-time `hydro.diversion.max_flow_m3s` — the entity read silently
        // drops any wired per-stage override (mirrors every sibling column family).
        let max_div = if hydro.filling.is_some() || suspended {
            0.0
        } else {
            ctx.resolved
                .bounds
                .hydro_bounds(h_idx, stage_idx)
                .max_diversion_m3s
                .unwrap_or(0.0)
        };
        for blk in 0..layout.n_blks {
            let col = layout.diversion_col(HydroSys::new(h_idx), BlockIdx::new(blk));
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
///
/// A genuinely anticipated thermal gets per-block bounds here but keeps objective
/// `0.0` — generation is priced once at the decision stage (`fill_anticipated_columns`);
/// pricing the delivery column too double-counts. "Genuinely anticipated" keys on
/// `layout.anticipated.anticipated_fishing_row_pos` (the fishing row's own gate), NOT a
/// static per-plant flag: a `K = 0` sub-stage-lead delivery has no fishing coupling and
/// prices normally, like an ordinary thermal.
///
/// A commissioning-dormant thermal (`commissioning_active == false`, keyed on `stage.id`)
/// forces BOTH bounds to `[0, 0]`: both must drop, or a `min_generation_mw` must-run floor
/// leaves the infeasible `[min > 0, 0]`. This generation column carries the operation-window
/// gate; the shifted gate (decision priced `K` stages early) lives on the decision column
/// in `fill_anticipated_columns`, not here.
pub(super) fn fill_thermal_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (t_idx, thermal) in ctx.thermals.iter().enumerate() {
        let active = commissioning_active(thermal.entry_stage_id, thermal.exit_stage_id, stage.id);
        let tb = ctx.resolved.bounds.thermal_bounds(t_idx, stage_idx);
        let marginal_cost_per_mwh = tb.cost_per_mwh;
        let is_anticipated =
            layout
                .anticipated_local_by_sys_pos
                .get(&t_idx)
                .is_some_and(|&local_idx| {
                    layout.anticipated.anticipated_fishing_row_pos[local_idx].is_some()
                });
        for blk in 0..layout.n_blks {
            let col =
                layout
                    .block_grid()
                    .flat(layout.equipment.thermal.start, t_idx, BlockIdx::new(blk));
            if active {
                bufs.col_lower[col] = tb.min_generation_mw;
                bufs.col_upper[col] = tb.max_generation_mw;
            } else {
                bufs.col_lower[col] = 0.0;
                bufs.col_upper[col] = 0.0;
            }
            if !is_anticipated {
                let block_hours = stage.blocks[blk].duration_hours;
                bufs.objective[col] = marginal_cost_per_mwh * block_hours;
            }
        }
    }
}

/// Anticipated-plant decision columns: state-out bound, decision bound, and
/// decision objective for the plant's single genuine decision this stage.
///
/// The decision column is bound/costed at ITS OWN delivery stage (`thermal_bounds`,
/// delivery hours/discount) — never the decision stage `stage_idx`, never
/// `stage_idx + constant` — and deposits into ring slot `delivery_stage - stage_idx - 1`,
/// the ring's direct delivery-distance mapping, never a `depth`-derived boundary (which
/// under-counts when pre-study occupancy coexists with an in-study decision at the same
/// stage). An inactive plant keeps its decision/state-out columns dormant `[0, 0]`.
///
/// Active (`is_anticipated_decision_active_for_delivery`) is evaluated at the decision's
/// OWN delivery stage; `delivery_stage == n_stages` is INACTIVE (strict gate) — pricing
/// it would create a cost-only column with no delivery LP. The `anticipated_state_out_def`
/// row is emitted iff the decision is active (lockstep: zero-bound iff no def row).
///
/// The decision objective is the present-value commit cost UNSCALED — the caller divides
/// every non-theta entry by `COST_SCALE_FACTOR`.
pub(super) fn fill_anticipated_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    let n_stages = ctx.resolved.bounds.n_stages();
    let n_ant = ctx.n_anticipated;
    let decision_start = layout.anticipated.col_anticipated_decision_start;
    let ring = super::entries::anticipated_ring(layout);

    for local_idx in 0..n_ant {
        let col = decision_start + local_idx;
        bufs.col_lower[col] = 0.0;
        bufs.col_upper[col] = 0.0;
    }

    let mut active_count = 0_usize;
    for local_idx in 0..n_ant {
        let point = layout
            .state
            .anticipated_resolution_for(AnticipatedLocal::new(local_idx), n_stages);
        let Some(delivery_stage) = point.genuine_decisions_at(stage_idx).next() else {
            continue;
        };
        let decision_col = decision_start + local_idx;
        debug_assert!(
            delivery_stage > stage_idx,
            "a genuine decision's delivery stage must be strictly after the decision \
             stage (K=0 self-delivery must already be excluded)"
        );
        let slot = delivery_stage - stage_idx - 1;
        debug_assert!(
            slot < layout.k_max,
            "delivery slot {slot} must be within the sized ring depth {}",
            layout.k_max
        );
        let state_out_col = ring.out_col(slot, local_idx);

        if layout.state.is_anticipated_decision_active_for_delivery(
            AnticipatedLocal::new(local_idx),
            delivery_stage,
            n_stages,
            &ctx.anticipated_windows,
            &ctx.study_stage_ids,
        ) {
            active_count += 1;
            let thermal_idx = ctx.anticipated_thermal_indices[local_idx];
            let tb = ctx
                .resolved
                .bounds
                .thermal_bounds(thermal_idx.get(), delivery_stage);

            bufs.col_lower[state_out_col] = f64::NEG_INFINITY;
            bufs.col_upper[state_out_col] = f64::INFINITY;

            bufs.col_lower[decision_col] = tb.min_generation_mw;
            bufs.col_upper[decision_col] = tb.max_generation_mw;

            let delivery_hours = ctx.total_hours_per_stage[delivery_stage];
            let d_factor = ctx.cumulative_discount_factors[delivery_stage];
            bufs.objective[decision_col] = tb.cost_per_mwh * delivery_hours * d_factor;
        }
    }
    debug_assert_eq!(
        active_count, layout.anticipated.n_anticipated_state_out_def_rows,
        "active state_out column count must match def-row count at stage {stage_idx}"
    );
}

/// Line columns per line per block (forward and reverse).
///
/// A commissioning-dormant line (`commissioning_active == false`, keyed on `stage.id`)
/// forces `col_upper` to `0` both directions; `col_lower` is already `0` (no
/// transmission floor), so only the cap drops.
fn fill_line_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (l_idx, line) in ctx.lines.iter().enumerate() {
        let active = commissioning_active(line.entry_stage_id, line.exit_stage_id, stage.id);
        let lb = ctx.resolved.bounds.line_bounds(l_idx, stage_idx);
        let lp = ctx.resolved.penalties.line_penalties(l_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let (df, rf) = ctx
                .resolved
                .resolved_exchange_factors
                .factors(l_idx, stage_idx, blk);
            let col_fwd = layout.line_fwd_col(LineSys::new(l_idx), BlockIdx::new(blk));
            let col_rev = layout.line_rev_col(LineSys::new(l_idx), BlockIdx::new(blk));
            if active {
                bufs.col_upper[col_fwd] = lb.direct_mw * df;
                bufs.col_upper[col_rev] = lb.reverse_mw * rf;
            } else {
                bufs.col_upper[col_fwd] = 0.0;
                bufs.col_upper[col_rev] = 0.0;
            }
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col_fwd] = lp.exchange_cost * block_hours;
            bufs.objective[col_rev] = lp.exchange_cost * block_hours;
        }
    }
}

/// Deficit and excess columns per bus per block.
///
/// The deficit region uses a uniform per-bus stride of `max_deficit_segments`
/// (column address owned by `deficit_col`). Buses with fewer segments leave the
/// trailing slots at the `[0, 0]` vec default, presolve-eliminated.
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
                let col_def = layout.deficit_col(b_idx, seg_idx, BlockIdx::new(blk));
                let block_hours = stage.blocks[blk].duration_hours;
                bufs.col_upper[col_def] = segment.depth_mw.unwrap_or(f64::INFINITY);
                bufs.objective[col_def] = segment.cost_per_mwh * block_hours;
            }
        }
        for blk in 0..layout.n_blks {
            let col_exc =
                layout
                    .block_grid()
                    .flat(layout.equipment.excess.start, b_idx, BlockIdx::new(blk));
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.col_upper[col_exc] = f64::INFINITY;
            bufs.objective[col_exc] = bp.excess_cost * block_hours;
        }
    }
}

/// Inflow non-negativity slack columns (`sigma_inf_h`), one per hydro. Bounds
/// `[0, +inf)` are the vec default; only the objective is written.
fn fill_inflow_slack_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    total_stage_hours: f64,
    bufs: &mut ColumnBufs<'_>,
) {
    if ctx.has_penalty {
        for h_idx in 0..layout.n_h {
            let col = layout.slack.inflow_slack.start + h_idx;
            let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
            bufs.objective[col] = hp.inflow_nonnegativity_cost * total_stage_hours;
        }
    }
}

/// FPHA generation columns (`g_{h,k}`): one per FPHA hydro per block, bounds
/// `[0, max_generation_mw]`, objective `0.0` (turbined cost is on the turbine column).
///
/// `identify_fpha_hydros` excludes a filling hydro from `fpha_hydro_indices` during
/// `PreFilling`/`Filling` — the single owner of "no generation while filling"; this
/// loop must NOT re-gate by phase (the branch would be dead).
fn fill_fpha_generation_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (local_idx, &h) in layout.fpha_hydro_indices.iter().enumerate() {
        let local_idx = FphaLocal::new(local_idx);
        let hb = ctx.resolved.bounds.hydro_bounds(h.get(), stage_idx);
        for blk in (0..layout.n_blks).map(BlockIdx::new) {
            let col = layout.generation_col(local_idx, blk);
            bufs.col_lower[col] = 0.0;
            bufs.col_upper[col] = hb.max_generation_mw;
        }
    }
}

/// Evaporation columns: one `EVAP_COLS_PER_HYDRO` triple (evaporation outflow,
/// `f_evap_plus`, `f_evap_minus`) per evaporation hydro per block.
///
/// The evaporation-outflow column is bounded symmetrically `[-q_max, +q_max]`, zero
/// objective. `f_evap_plus`/`f_evap_minus` are `[0, +inf)` and carry the directional
/// violation costs scaled by **that block's** `duration_hours`, not `total_stage_hours`
/// — the flow enters the water balance per block, so a stage-total factor inflates the
/// penalty `K`-fold at `K ≥ 2`.
fn fill_evaporation_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (local_idx, &h) in layout.evap_hydro_indices.iter().enumerate() {
        let local_idx = EvapLocal::new(local_idx);
        let (q_max_abs, hp) = match ctx.evaporation_models.model(h.get()) {
            EvaporationModel::Linearized { coefficients, .. } => {
                let coeff = &coefficients[stage_idx];
                let hb = ctx.resolved.bounds.hydro_bounds(h.get(), stage_idx);
                let q_max_abs = (coeff.intercept_m3s
                    + coeff.volume_slope_m3s_per_hm3 * hb.max_storage_hm3)
                    .abs()
                    * EVAPORATION_FLOW_SAFETY_MARGIN;
                (
                    q_max_abs,
                    ctx.resolved.penalties.hydro_penalties(h.get(), stage_idx),
                )
            }
            EvaporationModel::None => {
                debug_assert!(
                    false,
                    "evap_hydro_indices contains hydro {} but model is None",
                    h.get()
                );
                continue;
            }
        };
        for blk in 0..layout.n_blks {
            let col_evaporation_flow = layout.evap_flow_col(local_idx, BlockIdx::new(blk));
            let col_f_plus = layout.evap_f_plus_col(local_idx, BlockIdx::new(blk));
            let col_f_minus = layout.evap_f_minus_col(local_idx, BlockIdx::new(blk));
            // Signed: a negative outflow reads as net rainfall input (inflow).
            bufs.col_lower[col_evaporation_flow] = -q_max_abs;
            bufs.col_upper[col_evaporation_flow] = q_max_abs;
            bufs.col_lower[col_f_plus] = 0.0;
            bufs.col_upper[col_f_plus] = f64::INFINITY;
            bufs.col_lower[col_f_minus] = 0.0;
            bufs.col_upper[col_f_minus] = f64::INFINITY;
            // f_evap_plus = under-evaporation, f_evap_minus = over-evaporation.
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col_f_plus] = hp.evaporation_violation_neg_cost * block_hours;
            bufs.objective[col_f_minus] = hp.evaporation_violation_pos_cost * block_hours;
        }
    }
}

/// Withdrawal violation slack columns — neg (under-withdrawal) and pos
/// (over-withdrawal), one stage-level column per hydro per direction.
///
/// Realized withdrawal is `R = T - neg + pos`, `T = water_withdrawal_m3s` (signed:
/// `T > 0` removal, `T < 0` inter-basin return). `R` must NOT flip sign, so the
/// *under-delivery* direction (the one dragging `R` toward zero) is capped at `|T|`
/// while *over-delivery* is left `+∞`:
/// - `T > 0`: cap `neg ≤ |T|` (floors `R ≥ 0`); `pos` is `+∞`.
/// - `T < 0`: cap `pos ≤ |T|` (floors `R ≤ 0`); `neg` is `+∞`.
/// - `T = 0`: both `0` (presolve-eliminated).
fn fill_withdrawal_slack_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    total_stage_hours: f64,
    bufs: &mut ColumnBufs<'_>,
) {
    for h_idx in 0..layout.n_h {
        let col = layout.slack.withdrawal_slack_neg.start + h_idx;
        let hb = ctx.resolved.bounds.hydro_bounds(h_idx, stage_idx);
        let t = hb.water_withdrawal_m3s;
        bufs.col_upper[col] = if t > 0.0 {
            t
        } else if t < 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        bufs.objective[col] = hp.water_withdrawal_violation_neg_cost * total_stage_hours;
    }
    for h_idx in 0..layout.n_h {
        let col = layout.slack.withdrawal_slack_pos.start + h_idx;
        let hb = ctx.resolved.bounds.hydro_bounds(h_idx, stage_idx);
        let t = hb.water_withdrawal_m3s;
        bufs.col_upper[col] = if t > 0.0 {
            f64::INFINITY
        } else if t < 0.0 {
            -t
        } else {
            0.0
        };
        let hp = ctx.resolved.penalties.hydro_penalties(h_idx, stage_idx);
        bufs.objective[col] = hp.water_withdrawal_violation_pos_cost * total_stage_hours;
    }
}

/// One operational-violation slack family, addressing a disjoint `n_h * n_blks`
/// column range. Every activation predicate reads the **resolved per-stage** bound,
/// never the entity declaration — the declaration ignores per-stage overrides and
/// would silently mis-activate columns while still compiling.
#[derive(Clone, Copy)]
enum BlockSlackFamily {
    /// Active iff resolved `min_outflow_m3s > 0.0`.
    OutflowBelow,
    /// Active iff resolved `max_outflow_m3s.is_some()` — NOT `> 0.0`: a `Some(0.0)`
    /// cap still activates the column.
    OutflowAbove,
    /// Active iff resolved `min_turbined_m3s > 0.0`.
    TurbineBelow,
    /// Active iff resolved `min_generation_mw > 0.0`.
    GenerationBelow,
}

/// Operational violation slack columns: 4 families of `n_h * n_blks` columns, each
/// a disjoint range, so call order does not affect the result.
fn fill_operational_slack_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for family in [
        BlockSlackFamily::OutflowBelow,
        BlockSlackFamily::OutflowAbove,
        BlockSlackFamily::TurbineBelow,
        BlockSlackFamily::GenerationBelow,
    ] {
        fill_block_family(ctx, stage, stage_idx, layout, bufs, family);
    }
}

/// Fill one operational-violation slack family's `n_h * n_blks` columns; `col_lower`
/// stays at the `0.0` vec default.
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
                BlockSlackFamily::OutflowBelow => {
                    layout.outflow_below_col(HydroSys::new(h_idx), BlockIdx::new(blk))
                }
                BlockSlackFamily::OutflowAbove => {
                    layout.outflow_above_col(HydroSys::new(h_idx), BlockIdx::new(blk))
                }
                BlockSlackFamily::TurbineBelow => {
                    layout.turbine_below_col(HydroSys::new(h_idx), BlockIdx::new(blk))
                }
                BlockSlackFamily::GenerationBelow => {
                    layout.generation_below_col(HydroSys::new(h_idx), BlockIdx::new(blk))
                }
            };
            bufs.col_upper[col] = if active { f64::INFINITY } else { 0.0 };
            bufs.objective[col] = cost * stage.blocks[blk].duration_hours;
        }
    }
}

/// NCS generation columns: one per NCS per block, dense and system-indexed.
///
/// A commissioning-dormant NCS (`commissioning_active == false`) forces BOTH bounds to
/// `[0, 0]`: leaving the must-run lower bound at `upper > 0` would force generation from
/// a not-yet-commissioned source.
///
/// These template values govern only for non-stochastic NCS noise; with stochastic NCS,
/// `transform_ncs_noise` overwrites both bounds per scenario via `set_col_bounds`.
fn fill_ncs_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (ncs_sys_idx, ncs) in ctx.non_controllable_sources.iter().enumerate() {
        let active = commissioning_active(ncs.entry_stage_id, ncs.exit_stage_id, stage.id);
        let avail_gen = ctx
            .resolved
            .resolved_ncs_bounds
            .available_generation(ncs_sys_idx, stage_idx);
        for blk in 0..layout.n_blks {
            let col = layout.block_grid().flat(
                layout.equipment.col_ncs_start,
                ncs_sys_idx,
                BlockIdx::new(blk),
            );
            if active {
                let factor = ctx
                    .resolved
                    .resolved_ncs_factors
                    .factor(ncs_sys_idx, stage_idx, blk);
                let upper = avail_gen * factor;
                bufs.col_upper[col] = upper;
                bufs.col_lower[col] = if ncs.allow_curtailment { 0.0 } else { upper };
            } else {
                bufs.col_upper[col] = 0.0;
                bufs.col_lower[col] = 0.0;
            }
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col] = -ncs.curtailment_cost * block_hours;
        }
    }
}

/// Pumping-flow columns: one per station per block, dense and system-indexed.
/// Objective is zero — pumping's electrical cost enters through the bus load balance,
/// not here.
///
/// A commissioning-dormant station (`commissioning_active == false`) forces BOTH bounds
/// to `[0, 0]`: zeroing only `max` leaves the infeasible `[min > 0, 0]`.
pub(super) fn fill_pumping_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    for (p_sys, station) in ctx.pumping_stations.iter().enumerate() {
        let active = commissioning_active(station.entry_stage_id, station.exit_stage_id, stage.id);
        let pb = ctx.resolved.bounds.pumping_bounds(p_sys, stage_idx);
        for blk in (0..layout.n_blks).map(BlockIdx::new) {
            let col = layout
                .block_grid()
                .flat(layout.equipment.col_pumping_start, p_sys, blk);
            if active {
                bufs.col_lower[col] = pb.min_flow_m3s;
                bufs.col_upper[col] = pb.max_flow_m3s;
            } else {
                bufs.col_lower[col] = 0.0;
                bufs.col_upper[col] = 0.0;
            }
        }
    }
}

/// Energy-contract columns. The family base (`col_contract_import_start` /
/// `col_contract_export_start`) is addressed by the per-family slot from
/// [`contract_family_slot`](crate::generic_constraints::contract_family_slot) — the
/// single owner the load-balance fill and the resolver also share — not by `c_sys`.
///
/// A commissioning-dormant contract has BOTH bounds forced to `[0, 0]`: zeroing only
/// `max` would leave the infeasible `[min > 0, 0]` for a take-or-pay floor.
///
/// The objective is `price_per_mwh * block_hours`, written UNSCALED and UNNEGATED for
/// both families and regardless of commissioning — the stored price sign carries
/// direction (import `> 0` cost, export `< 0` revenue) and the prescaling pass owns
/// `col_scale`.
fn fill_contract_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    let grid = layout.block_grid();
    for (c_sys, contract) in ctx.contracts.iter().enumerate() {
        let active =
            commissioning_active(contract.entry_stage_id, contract.exit_stage_id, stage.id);
        let cb = ctx.resolved.bounds.contract_bounds(c_sys, stage_idx);
        let (contract_type, family_slot) = contract_family_slot(ctx.contracts, c_sys);
        let (base, family_count) = match contract_type {
            ContractType::Import => (
                layout.equipment.col_contract_import_start,
                layout.equipment.n_contract_import,
            ),
            ContractType::Export => (
                layout.equipment.col_contract_export_start,
                layout.equipment.n_contract_export,
            ),
        };
        debug_assert!(
            family_slot < family_count,
            "contract family slot {family_slot} out of range {family_count} at stage {stage_idx}"
        );
        for blk in 0..layout.n_blks {
            let col = grid.flat(base, family_slot, BlockIdx::new(blk));
            if active {
                bufs.col_lower[col] = cb.min_mw;
                bufs.col_upper[col] = cb.max_mw;
            } else {
                bufs.col_lower[col] = 0.0;
                bufs.col_upper[col] = 0.0;
            }
            let block_hours = stage.blocks[blk].duration_hours;
            bufs.objective[col] = cb.price_per_mwh * block_hours;
        }
    }
}

/// Per-stage `σ_fill`-target slack columns: one stage-level slack per Filling-phase
/// filling hydro. `[0, +∞)`, objective is the RESOLVED `filling_target_violation_cost`,
/// written UNSCALED (the caller divides non-theta entries by `COST_SCALE_FACTOR`).
///
/// CRITICAL — the cost is NOT multiplied by stage hours. `σ_fill` is a
/// STORAGE-VOLUME slack (hm³) and the cost is $/hm³, so `σ_fill · cost` is already
/// $. The wrong-but-compiling alternative — copying the `* total_stage_hours`
/// factor the flow/power-RATE slacks carry — is a $·h/hm³ units error that lets the
/// optimizer violate the target ~744× too cheaply. (`σ^{v-}` shares this convention.)
fn fill_filling_target_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    let col_start = layout.filling.col_filling_target_start;
    for (local_idx, &h) in layout
        .filling
        .filling_target_hydro_indices
        .iter()
        .enumerate()
    {
        let local_idx = FillingTargetLocal::new(local_idx);
        let col = col_start + local_idx.get();
        let hp = ctx.resolved.penalties.hydro_penalties(h.get(), stage_idx);
        bufs.col_lower[col] = 0.0;
        bufs.col_upper[col] = f64::INFINITY;
        bufs.objective[col] = hp.filling_target_violation_cost;
    }
}

/// Soft `σ^{v-}` operating-floor slack columns: one stage-level slack per
/// Operating-phase filling hydro. `[0, +∞)`, objective is the RESOLVED
/// `storage_violation_below_cost`, written UNSCALED with the SAME no-hours,
/// $/hm³ units contract as `fill_filling_target_columns` (`σ_fill`).
///
/// DISTINCT from `σ_fill`: `σ^{v-}` fires at EVERY Operating stage, `σ_fill` at
/// every Filling stage; separate columns, costs, and stage scopes (never overlap).
fn fill_filled_min_storage_floor_columns(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    bufs: &mut ColumnBufs<'_>,
) {
    let col_start = layout.filling.col_filled_min_storage_floor_start;
    for (local_idx, &h) in layout
        .filling
        .filled_min_storage_floor_hydro_indices
        .iter()
        .enumerate()
    {
        let local_idx = FloorLocal::new(local_idx);
        let col = col_start + local_idx.get();
        let hp = ctx.resolved.penalties.hydro_penalties(h.get(), stage_idx);
        bufs.col_lower[col] = 0.0;
        bufs.col_upper[col] = f64::INFINITY;
        bufs.objective[col] = hp.storage_violation_below_cost;
    }
}

/// Z-inflow columns: free variables for realized total inflow per hydro.
fn fill_z_inflow_columns(layout: &StageLayout, bufs: &mut ColumnBufs<'_>) {
    for h_idx in 0..layout.n_h {
        let col = layout.col_z_inflow_start() + h_idx;
        bufs.col_lower[col] = f64::NEG_INFINITY;
        bufs.col_upper[col] = f64::INFINITY;
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::needless_range_loop,
    clippy::similar_names
)]
mod interior_storage_bound_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::{
        Block, BlockMode, BoundsCountsSpec, BoundsDefaults, BusStagePenalties, CascadeTopology,
        ContractStageBounds, EntityId, Hydro, HydroStageBounds, HydroStagePenalties,
        LineStageBounds, LineStagePenalties, NcsStagePenalties, NoiseMethod, PenaltiesCountsSpec,
        PenaltiesDefaults, PumpingStageBounds, ResolvedBounds, ResolvedExchangeFactors,
        ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds,
        ResolvedNcsFactors, ResolvedPenalties, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, ProductionModelSet, ResolvedProductionModel,
    };
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::state_layout_for;
    use super::{ColumnBufs, StageLayout, TemplateBuildCtx, fill_storage_columns};
    use crate::indexer::{Boundary, HydroSys};

    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;
    const N_BLKS: usize = 3;
    const MIN_STORAGE_HM3: f64 = 50.0;
    const MAX_STORAGE_HM3: f64 = 175.0;

    /// One Operating run-of-river hydro (`ConstantProductivity`, no filling, no
    /// commissioning window ⇒ `floor_off == false`), so the endpoint outgoing
    /// storage column takes the hard `[MIN_STORAGE_HM3, MAX_STORAGE_HM3]` floor and
    /// interior == endpoint is the unambiguous expectation. No FPHA/evaporation, so
    /// the layout reserves only the structural storage families.
    fn operating_hydro() -> Hydro {
        Hydro {
            id: EntityId(1),
            name: "H1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: MIN_STORAGE_HM3,
            max_storage_hm3: MAX_STORAGE_HM3,
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
            diversion: None,
            filling: None,
            penalties: super::super::test_support::zero_hydro_penalties(),
        }
    }

    /// `ResolvedBounds` for one hydro carrying the distinct `[MIN, MAX]` storage
    /// range so the assertions bite (a wrong cell read would not equal MIN/MAX).
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
                    min_storage_hm3: MIN_STORAGE_HM3,
                    max_storage_hm3: MAX_STORAGE_HM3,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 100.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 250.0,
                    max_diversion_m3s: None,
                    filling_min_rate_m3s: 0.0,
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

    /// All-zero penalties for one hydro across `N_STAGES` stages so no fixture-side
    /// cost contaminates the storage objective/scale assertions, and so the full
    /// template build's penalty reads land on a populated cell.
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

    /// A `Stage` with `N_BLKS` equal-duration blocks under `block_mode`.
    fn stage_with_blocks(block_mode: BlockMode) -> Stage {
        Stage {
            index: STAGE_IDX,
            id: STAGE_IDX as i32,
            start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: (0..N_BLKS)
                .map(|index| Block {
                    index,
                    name: format!("BLK{index}"),
                    duration_hours: 248.0,
                })
                .collect(),
            block_mode,
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

    /// Owns the borrow targets for a one-hydro `TemplateBuildCtx`.
    struct InteriorStorageFixtures {
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

    impl InteriorStorageFixtures {
        fn new() -> Self {
            let hydros = vec![operating_hydro()];
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
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
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
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: vec![],
                has_penalty: false,
                cumulative_discount_factors: vec![1.0],
                total_hours_per_stage: vec![744.0],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// Run `fill_storage_columns` against raw, unscaled buffers for `stage`,
    /// returning the bound/objective buffers plus the resolved storage-column
    /// offsets by value (the borrowed `StateLayout` cannot escape).
    fn run_fill(fixtures: &InteriorStorageFixtures, stage: &Stage) -> RawFill {
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, stage, STAGE_IDX);
        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_storage_columns(&ctx, stage, STAGE_IDX, &layout, &mut bufs);
        // The actual interior columns are the `storage_internal` range members
        // (empty in parallel mode and at K = 1); `block_storage_col(0, k)` for
        // interior `k` resolves into this range only in chronological K ≥ 2.
        let interior: Vec<usize> = layout.equipment.storage_internal.clone().collect();
        RawFill {
            col_lower,
            col_upper,
            objective,
            endpoint: layout.block_storage_col(HydroSys::new(0), Boundary::Outgoing),
            interior,
            storage_internal_empty: layout.equipment.storage_internal.is_empty(),
        }
    }

    /// Raw `fill_storage_columns` output plus the storage column offsets the
    /// assertions read.
    struct RawFill {
        col_lower: Vec<f64>,
        col_upper: Vec<f64>,
        objective: Vec<f64>,
        endpoint: usize,
        interior: Vec<usize>,
        storage_internal_empty: bool,
    }

    /// Interior `Sᵏ` columns inherit the endpoint outgoing-storage bounds, carry a
    /// `0.0` objective, and take the matrix-derived empty-column scale `1.0` while
    /// they carry no row coefficients. Parallel mode produces no interior columns,
    /// with storage bounds and objective unchanged.
    #[test]
    fn interior_storage_columns_inherit_stage_bounds_objective_scale() {
        let fixtures = InteriorStorageFixtures::new();

        // Bounds + objective: raw, unscaled buffers in chronological K = 3.
        let chrono = run_fill(&fixtures, &stage_with_blocks(BlockMode::Chronological));
        assert!(
            !chrono.storage_internal_empty,
            "chronological K=3 must reserve interior storage columns"
        );
        assert_eq!(chrono.interior.len(), N_BLKS - 1, "K - 1 interior columns");
        let endpoint_lower = chrono.col_lower[chrono.endpoint];
        let endpoint_upper = chrono.col_upper[chrono.endpoint];
        assert_eq!(
            endpoint_lower, MIN_STORAGE_HM3,
            "endpoint floor is the hard min"
        );
        assert_eq!(endpoint_upper, MAX_STORAGE_HM3, "endpoint cap is the max");
        for &col in &chrono.interior {
            assert_eq!(
                chrono.col_lower[col], endpoint_lower,
                "interior col {col} lower bound must equal the endpoint storage lower bound"
            );
            assert_eq!(
                chrono.col_upper[col], endpoint_upper,
                "interior col {col} upper bound must equal the endpoint storage upper bound"
            );
            assert_eq!(
                chrono.objective[col], 0.0,
                "interior col {col} objective must stay 0.0"
            );
        }

        // Scale: the matrix-derived scale of a structurally-empty interior column is
        // 1.0 while it carries no row coefficients (later water-balance/FPHA fills add
        // them). Build the full chronological template and run the production scale
        // computation on its CSC.
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let chrono_stage = stage_with_blocks(BlockMode::Chronological);
        let layout = StageLayout::new(&ctx, &state, &chrono_stage, STAGE_IDX);
        let template = super::super::template::build_single_stage_template(
            &ctx,
            &state,
            &chrono_stage,
            STAGE_IDX,
        )
        .template;
        let col_scale = super::super::compute_col_scale(
            template.num_cols,
            &template.col_starts,
            &template.values,
        );
        for k in 1..layout.n_blks {
            let col = layout.block_storage_col(HydroSys::new(0), Boundary::Interior(k));
            assert_eq!(
                col_scale[col], 1.0,
                "interior col {col} scale must be the empty-column scale 1.0 while it carries no row coefficients"
            );
        }

        // Parallel identity: the interior loop's range is empty in parallel mode, so
        // it touches no column. Two independent parallel builds must therefore be
        // bit-for-bit identical in bounds, objective, and dense matrix — and neither
        // reserves interior columns. (A change perturbing the inert loop into the
        // parallel column block would break this dense comparison.)
        let parallel = run_fill(&fixtures, &stage_with_blocks(BlockMode::Parallel));
        assert!(
            parallel.storage_internal_empty,
            "parallel storage_internal must be empty (no interior columns)"
        );
        assert_eq!(
            parallel.interior,
            Vec::<usize>::new(),
            "parallel mode resolves no interior storage columns"
        );

        let build_parallel = || {
            let par_ctx = fixtures.make_ctx();
            let par_state = state_layout_for(&par_ctx);
            let parallel_stage = stage_with_blocks(BlockMode::Parallel);
            super::super::template::build_single_stage_template(
                &par_ctx,
                &par_state,
                &parallel_stage,
                STAGE_IDX,
            )
            .template
        };
        let tpl_a = build_parallel();
        let tpl_b = build_parallel();
        assert_eq!(tpl_a.num_cols, tpl_b.num_cols, "parallel num_cols stable");
        let dense_a = csc_to_dense(&tpl_a);
        let dense_b = csc_to_dense(&tpl_b);
        for j in 0..tpl_a.num_cols {
            assert_eq!(
                tpl_a.col_lower[j].to_bits(),
                tpl_b.col_lower[j].to_bits(),
                "parallel col_lower differs at col {j}"
            );
            assert_eq!(
                tpl_a.col_upper[j].to_bits(),
                tpl_b.col_upper[j].to_bits(),
                "parallel col_upper differs at col {j}"
            );
            assert_eq!(
                tpl_a.objective[j].to_bits(),
                tpl_b.objective[j].to_bits(),
                "parallel objective differs at col {j}"
            );
        }
        for i in 0..tpl_a.num_rows {
            for j in 0..tpl_a.num_cols {
                assert_eq!(
                    dense_a[i][j].to_bits(),
                    dense_b[i][j].to_bits(),
                    "parallel matrix differs at row {i} col {j}"
                );
            }
        }

        let (endpoint, par_storage_internal_empty) = {
            let par_ctx = fixtures.make_ctx();
            let par_state = state_layout_for(&par_ctx);
            let stage = stage_with_blocks(BlockMode::Parallel);
            let l = StageLayout::new(&par_ctx, &par_state, &stage, STAGE_IDX);
            (
                l.block_storage_col(HydroSys::new(0), Boundary::Outgoing),
                l.equipment.storage_internal.is_empty(),
            )
        };
        assert!(
            par_storage_internal_empty,
            "parallel build reserves no interior storage columns"
        );
        assert_eq!(
            tpl_a.col_lower[endpoint], MIN_STORAGE_HM3,
            "parallel endpoint storage lower bound unchanged"
        );
        assert_eq!(
            tpl_a.col_upper[endpoint], MAX_STORAGE_HM3,
            "parallel endpoint storage upper bound unchanged"
        );
        assert_eq!(
            tpl_a.objective[endpoint], 0.0,
            "parallel endpoint storage objective unchanged"
        );
    }

    /// Expand a CSC `StageTemplate` to a dense `Vec<Vec<f64>>` (mirrors the
    /// `template/tests.rs` dense-comparison helper).
    #[allow(clippy::cast_sign_loss)]
    fn csc_to_dense(tpl: &cobre_solver::StageTemplate) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0_f64; tpl.num_cols]; tpl.num_rows];
        for j in 0..tpl.num_cols {
            let start = tpl.col_starts[j] as usize;
            let end = tpl.col_starts[j + 1] as usize;
            for nz in start..end {
                let row = tpl.row_indices[nz] as usize;
                dense[row][j] = tpl.values[nz];
            }
        }
        dense
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
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
    use cobre_core::{
        BusStagePenalties, LineStagePenalties, NcsStagePenalties, ResolvedGenericConstraintBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, ProductionModelSet, ResolvedProductionModel,
    };
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::{state_layout_for, two_block_stage, zero_hydro_penalties};
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
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
            travel_time_hours: None,
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
                    filling_min_rate_m3s: 0.0,
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
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
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
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: vec![],
                has_penalty: false,
                cumulative_discount_factors: vec![1.0],
                total_hours_per_stage: vec![744.0],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// Run `fill_diversion_columns` against the fixture and return `col_upper`
    /// plus the two layout offsets the assertions read.
    ///
    /// Returns the layout's `(n_blks, col_diversion_start)` by value rather than
    /// the `StageLayout` itself: the layout borrows the function-local
    /// `StateLayout`, so it cannot escape — the caller only needs these two
    /// offsets to index `col_upper`.
    fn run_fill(fixtures: &DivFixtures) -> (Vec<f64>, usize, usize) {
        let stage = two_block_stage(STAGE_IDX, [372.0, 372.0]);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_diversion_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);
        (col_upper, layout.n_blks, layout.equipment.diversion.start)
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

        let (col_upper, n_blks, col_diversion_start) = run_fill(&fixtures);
        for blk in 0..n_blks {
            let col = col_diversion_start + blk;
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

        let (col_upper, n_blks, col_diversion_start) = run_fill(&fixtures);
        for blk in 0..n_blks {
            let col = col_diversion_start + blk;
            assert_eq!(
                col_upper[col], DECLARATION_MAX_FLOW_M3S,
                "blk {blk}: col_upper[{col}] must equal the declaration default \
                 {DECLARATION_MAX_FLOW_M3S} when no override tightens it"
            );
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod filling_phase_gating_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::hydro::{FillingConfig, HydroGenerationModel};
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, CascadeTopology, ContractStageBounds,
        EntityId, Hydro, HydroStageBounds, HydroStagePenalties, LineStageBounds,
        LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
        PumpingStageBounds, ResolvedBounds, ResolvedExchangeFactors,
        ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds,
        ResolvedNcsFactors, ResolvedPenalties, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, FphaPlane, ProductionModelSet,
        ResolvedProductionModel,
    };
    use crate::indexer::{BlockIdx, FphaLocal, HydroSys};
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::{state_layout_for, two_block_stage, zero_hydro_penalties};
    use super::{
        ColumnBufs, StageLayout, TemplateBuildCtx, fill_diversion_columns,
        fill_fpha_generation_columns, fill_spillage_columns, fill_turbine_columns,
    };

    const MAX_TURBINED_M3S: f64 = 100.0;
    const MAX_GENERATION_MW: f64 = 250.0;
    const MAX_DIVERSION_M3S: f64 = 60.0;
    const START_STAGE_ID: i32 = 2;
    const ENTRY_STAGE_ID: i32 = 4;
    // One bounds row suffices: the phase is keyed on `stage.id`, which the test
    // varies independently of the resolved-bound stage index (always 0).
    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;

    // Three representative stage ids, one per phase, for the
    // `start_stage_id = 2`, `entry_stage_id = 4` window:
    //   id 1 < start            -> PreFilling
    //   start <= id 3 < entry   -> Filling
    //   id 4 >= entry           -> Operating
    const PREFILLING_ID: i32 = 1;
    const FILLING_ID: i32 = 3;
    const OPERATING_ID: i32 = 4;

    /// One hydro, optionally filling, optionally FPHA. `ConstantProductivity`
    /// reserves no FPHA generation column; `Fpha` reserves one per block so the
    /// generation-column gate can be exercised.
    fn hydro(filling: Option<FillingConfig>, entry: Option<i32>, fpha: bool) -> Hydro {
        Hydro {
            id: EntityId(1),
            name: "H1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
            travel_time_hours: None,
            entry_stage_id: entry,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: if fpha {
                HydroGenerationModel::Fpha
            } else {
                HydroGenerationModel::ConstantProductivity
            },
            min_turbined_m3s: 0.0,
            max_turbined_m3s: MAX_TURBINED_M3S,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: MAX_GENERATION_MW,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling,
            penalties: zero_hydro_penalties(),
        }
    }

    fn bounds_one_hydro() -> ResolvedBounds {
        let mut bounds = ResolvedBounds::new(
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
                    max_turbined_m3s: MAX_TURBINED_M3S,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: MAX_GENERATION_MW,
                    max_diversion_m3s: Some(MAX_DIVERSION_M3S),
                    filling_min_rate_m3s: 0.0,
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
        bounds.hydro_bounds_mut(0, STAGE_IDX).max_diversion_m3s = Some(MAX_DIVERSION_M3S);
        bounds
    }

    /// One-hydro all-zero penalties table sized to `N_STAGES`. A properly sized
    /// table (not `ResolvedPenalties::empty()`) is required: the column fills read
    /// `hydro_penalties(0, 0)`, which indexes into the table and would panic on an
    /// empty one.
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
                hydro: zero_hydro_stage_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        )
    }

    /// All-zero `HydroStagePenalties` — the per-stage resolved analogue of
    /// `zero_hydro_penalties` (which produces a declaration-time `HydroPenalties`).
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

    /// Owns the borrow targets for a one-hydro `TemplateBuildCtx`.
    struct Fixtures {
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

    impl Fixtures {
        fn new(filling: Option<FillingConfig>, entry: Option<i32>, fpha: bool) -> Self {
            let hydros = vec![hydro(filling, entry, fpha)];
            let cascade = CascadeTopology::build(&hydros);
            let model = if fpha {
                ResolvedProductionModel::Fpha {
                    planes: vec![FphaPlane {
                        intercept: 0.0,
                        gamma_v: 0.0,
                        gamma_q: 0.0,
                        gamma_s: 0.0,
                    }],
                }
            } else {
                ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }
            };
            Self {
                par_lp: PrecomputedPar::default(),
                hydros,
                cascade,
                bounds: bounds_one_hydro(),
                penalties: penalties_one_hydro(),
                production_models: ProductionModelSet::new(
                    vec![vec![model; N_STAGES]],
                    1,
                    N_STAGES,
                ),
                evaporation_models: EvaporationModelSet::new(vec![EvaporationModel::None]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
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
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: vec![],
                has_penalty: false,
                cumulative_discount_factors: vec![1.0],
                total_hours_per_stage: vec![744.0],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// Fresh `[0, +INF]`-initialised column buffers sized to `layout.num_cols`.
    fn fresh_bufs(num_cols: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        (
            vec![0.0_f64; num_cols],
            vec![f64::INFINITY; num_cols],
            vec![0.0_f64; num_cols],
        )
    }

    /// Run the three column fills against a fixture at the given `stage_id`,
    /// returning `(col_lower, col_upper)` and the column offsets the assertions
    /// read. The layout and resolved-bound lookups use `STAGE_IDX`; the phase is
    /// keyed on `stage_id` alone, so building the stage at `stage_id` (its `id`)
    /// while pinning the resolved-bound index lets one bounds row serve all phases.
    fn run_fills(fixtures: &Fixtures, stage_id: i32) -> (Vec<f64>, Vec<f64>, [usize; 3]) {
        let stage_index = usize::try_from(stage_id).expect("test stage ids are non-negative");
        let stage = two_block_stage(stage_index, [372.0, 372.0]);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let (mut col_lower, mut col_upper, mut objective) = fresh_bufs(layout.num_cols);
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_turbine_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);
        fill_diversion_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);
        fill_fpha_generation_columns(&ctx, STAGE_IDX, &layout, &mut bufs);
        let offsets = [
            layout.turbine_col(HydroSys::new(0), BlockIdx::new(0)),
            layout.equipment.diversion.start,
            // FPHA-local index 0 (the sole FPHA hydro); for a non-FPHA fixture
            // there is no generation column, so callers must not read this slot.
            if layout.fpha_hydro_indices.is_empty() {
                usize::MAX
            } else {
                layout.generation_col(FphaLocal::new(0), BlockIdx::new(0))
            },
        ];
        (col_lower, col_upper, offsets)
    }

    /// Run `fill_spillage_columns` against the fixture at `stage_id`, returning
    /// `col_upper` and the spillage column start. Isolated like `run_storage_fill`:
    /// the spillage freeze is independent of the turbine/diversion/generation gates,
    /// so it is exercised on its own.
    fn run_spillage_fill(fixtures: &Fixtures, stage_id: i32) -> (Vec<f64>, usize, usize) {
        let stage_index = usize::try_from(stage_id).expect("test stage ids are non-negative");
        let stage = two_block_stage(stage_index, [372.0, 372.0]);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let (mut col_lower, mut col_upper, mut objective) = fresh_bufs(layout.num_cols);
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_spillage_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);
        (col_upper, layout.equipment.spillage.start, layout.n_blks)
    }

    fn filling_config() -> FillingConfig {
        FillingConfig {
            start_stage_id: START_STAGE_ID,
            filling_min_rate_m3s: 0.0,
        }
    }

    /// A filling hydro's spillage is pinned `[0, 0]` in `PreFilling` ONLY (no dam yet
    /// to spill from), but stays FREE `[0, +∞)` in `Filling` (a real impounding
    /// reservoir can spill over-dam excess) and `Operating`. The forbidden
    /// alternative — extending the freeze to `Filling` — removes that legitimate
    /// relief valve (D40).
    #[test]
    fn filling_hydro_spillage_frozen_in_prefilling_free_in_filling_and_operating() {
        let fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
        let (upper_pre, spill_start, n_blks) = run_spillage_fill(&fixtures, PREFILLING_ID);
        for blk in 0..n_blks {
            assert_eq!(
                upper_pre[spill_start + blk],
                0.0,
                "spillage col_upper frozen [0,0] in PreFilling, blk {blk}"
            );
        }
        for stage_id in [FILLING_ID, OPERATING_ID] {
            let (upper, start, n) = run_spillage_fill(&fixtures, stage_id);
            for blk in 0..n {
                assert_eq!(
                    upper[start + blk],
                    f64::INFINITY,
                    "spillage col_upper free [0,+∞) at stage {stage_id}, blk {blk}"
                );
            }
        }
    }

    /// A commissioning-dormant non-filling hydro (`filling = None`, `entry`) has its
    /// spillage pinned `[0, 0]` while `PreFilling` (the un-built dam spills nothing),
    /// regaining free spillage from `entry` onward (`Operating`).
    #[test]
    fn dormant_non_filling_hydro_spillage_frozen_before_entry_free_after() {
        let fixtures = Fixtures::new(None, Some(ENTRY_STAGE_ID), false);
        // With filling = None, both ids < entry are PreFilling (no Filling phase
        // exists for a non-filling hydro): FILLING_ID here is just a second dormant id.
        for stage_id in [PREFILLING_ID, FILLING_ID] {
            let (upper, start, n_blks) = run_spillage_fill(&fixtures, stage_id);
            for blk in 0..n_blks {
                assert_eq!(
                    upper[start + blk],
                    0.0,
                    "dormant spillage col_upper frozen [0,0] at PreFilling stage {stage_id}, blk {blk}"
                );
            }
        }
        let (upper, start, n_blks) = run_spillage_fill(&fixtures, OPERATING_ID);
        for blk in 0..n_blks {
            assert_eq!(
                upper[start + blk],
                f64::INFINITY,
                "spillage col_upper free [0,+∞) once Operating, blk {blk}"
            );
        }
    }

    /// A non-filling hydro with no commissioning window is `Operating` at every stage:
    /// its spillage stays free `[0, +∞)` at every stage id (parity-neutral — the
    /// freeze never fires).
    #[test]
    fn non_filling_hydro_spillage_free_at_every_stage() {
        let fixtures = Fixtures::new(None, None, false);
        for stage_id in [PREFILLING_ID, FILLING_ID, OPERATING_ID] {
            let (upper, start, n_blks) = run_spillage_fill(&fixtures, stage_id);
            for blk in 0..n_blks {
                assert_eq!(
                    upper[start + blk],
                    f64::INFINITY,
                    "non-filling spillage col_upper free at stage {stage_id}, blk {blk}"
                );
            }
        }
    }

    /// A filling FPHA hydro pins its turbine column to `[0, 0]` in `PreFilling`
    /// and `Filling` (turbines not installed). Both bounds drop — zeroing only
    /// `col_upper` would leave any `col_lower` floor live and make the LP
    /// infeasible during filling.
    ///
    /// The FPHA generation column has **no bound to zero** in these phases: the
    /// per-stage FPHA row exclusion (`identify_fpha_hydros`) drops the filling
    /// hydro from `fpha_hydro_indices`, so no generation column is allocated at
    /// all — the dense FPHA-local block omits it rather than emitting an open
    /// `[0, max]` column. `run_fills` therefore reports `usize::MAX` for the
    /// generation slot here (the no-generation-column sentinel), which is why
    /// this test asserts only the turbine bound. Generation re-enters the dense
    /// block once `Operating` — covered by
    /// `filling_hydro_turbine_and_generation_normal_in_operating`.
    #[test]
    fn filling_hydro_turbine_zeroed_and_no_generation_column_before_entry() {
        let fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), true);
        for stage_id in [PREFILLING_ID, FILLING_ID] {
            let (lower, upper, [turb, _div, gen_col]) = run_fills(&fixtures, stage_id);
            assert_eq!(lower[turb], 0.0, "turbine col_lower at stage {stage_id}");
            assert_eq!(upper[turb], 0.0, "turbine col_upper at stage {stage_id}");
            assert_eq!(
                gen_col,
                usize::MAX,
                "filling hydro has no FPHA generation column during filling (stage {stage_id})"
            );
        }
    }

    /// In Operating, a filling hydro's turbine and generation columns return to
    /// their normal operating bounds — only diversion stays gated all-phases.
    #[test]
    fn filling_hydro_turbine_and_generation_normal_in_operating() {
        let fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), true);
        let (lower, upper, [turb, _div, gen_col]) = run_fills(&fixtures, OPERATING_ID);
        assert_eq!(lower[turb], 0.0, "turbine col_lower");
        assert_eq!(upper[turb], MAX_TURBINED_M3S, "turbine col_upper");
        assert_eq!(lower[gen_col], 0.0, "generation col_lower");
        assert_eq!(upper[gen_col], MAX_GENERATION_MW, "generation col_upper");
    }

    /// A filling hydro's diversion column is `[0, 0]` in ALL three phases — gated
    /// on `filling.is_some()`, not on the phase, so entry does not re-enable it.
    #[test]
    fn filling_hydro_diversion_zeroed_in_all_phases() {
        let fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), true);
        for stage_id in [PREFILLING_ID, FILLING_ID, OPERATING_ID] {
            let (lower, upper, [_turb, div, _gen_col]) = run_fills(&fixtures, stage_id);
            for blk in 0..2 {
                let col = div + blk;
                assert_eq!(
                    lower[col], 0.0,
                    "diversion col_lower blk {blk} at {stage_id}"
                );
                assert_eq!(
                    upper[col], 0.0,
                    "diversion col_upper blk {blk} at {stage_id}"
                );
            }
        }
    }

    /// A non-filling FPHA hydro is `Operating` at every stage: turbine,
    /// generation, and diversion columns keep their normal bounds at every
    /// stage id (the parity-neutrality contract — all three gates no-op).
    #[test]
    fn non_filling_hydro_unchanged_at_every_stage() {
        let fixtures = Fixtures::new(None, None, true);
        for stage_id in [PREFILLING_ID, FILLING_ID, OPERATING_ID] {
            let (lower, upper, [turb, div, gen_col]) = run_fills(&fixtures, stage_id);
            assert_eq!(lower[turb], 0.0, "turbine col_lower at {stage_id}");
            assert_eq!(
                upper[turb], MAX_TURBINED_M3S,
                "turbine col_upper at {stage_id}"
            );
            assert_eq!(lower[gen_col], 0.0, "generation col_lower at {stage_id}");
            assert_eq!(
                upper[gen_col], MAX_GENERATION_MW,
                "generation col_upper at {stage_id}"
            );
            for blk in 0..2 {
                let col = div + blk;
                assert_eq!(
                    lower[col], 0.0,
                    "diversion col_lower blk {blk} at {stage_id}"
                );
                assert_eq!(
                    upper[col], MAX_DIVERSION_M3S,
                    "diversion col_upper blk {blk} at {stage_id}"
                );
            }
        }
    }

    /// A commissioning-dormant non-filling hydro (`filling = None`,
    /// `entry = Some(4)`) is `PreFilling` before entry: turbine and diversion
    /// columns are `[0, 0]`, and the FPHA generation column is omitted from the
    /// dense block (`identify_fpha_hydros` drops it), exactly like a filling hydro's
    /// `PreFilling` stage.
    #[test]
    fn dormant_non_filling_hydro_zeroed_before_entry() {
        let fixtures = Fixtures::new(None, Some(ENTRY_STAGE_ID), true);
        for stage_id in [PREFILLING_ID, FILLING_ID] {
            let (lower, upper, [turb, div, gen_col]) = run_fills(&fixtures, stage_id);
            assert_eq!(lower[turb], 0.0, "turbine col_lower at stage {stage_id}");
            assert_eq!(upper[turb], 0.0, "turbine col_upper at stage {stage_id}");
            for blk in 0..2 {
                let col = div + blk;
                assert_eq!(
                    lower[col], 0.0,
                    "diversion col_lower blk {blk} at {stage_id}"
                );
                assert_eq!(
                    upper[col], 0.0,
                    "dormant diversion col_upper blk {blk} at {stage_id}"
                );
            }
            assert_eq!(
                gen_col,
                usize::MAX,
                "dormant non-filling hydro has no FPHA generation column before entry (stage {stage_id})"
            );
        }
    }

    /// From `entry` onward a commissioning-dormant non-filling hydro is `Operating`
    /// with NO intervening `Filling` phase: turbine, generation, and diversion
    /// return to their normal bounds at the first commissioned stage.
    #[test]
    fn dormant_non_filling_hydro_normal_from_entry() {
        let fixtures = Fixtures::new(None, Some(ENTRY_STAGE_ID), true);
        let (lower, upper, [turb, div, gen_col]) = run_fills(&fixtures, OPERATING_ID);
        assert_eq!(lower[turb], 0.0, "turbine col_lower");
        assert_eq!(upper[turb], MAX_TURBINED_M3S, "turbine col_upper");
        assert_eq!(lower[gen_col], 0.0, "generation col_lower");
        assert_eq!(upper[gen_col], MAX_GENERATION_MW, "generation col_upper");
        for blk in 0..2 {
            let col = div + blk;
            assert_eq!(
                upper[col], MAX_DIVERSION_M3S,
                "diversion col_upper blk {blk}"
            );
        }
    }

    // A non-zero dead volume so the storage-floor relax (floor → 0) is observable
    // against the hard `min_storage` floor.
    const MIN_STORAGE_HM3: f64 = 50.0;
    const MAX_STORAGE_HM3: f64 = 200.0;

    /// Run `fill_storage_columns` against the fixture at `stage_id` with the
    /// resolved dead volume overridden to `MIN_STORAGE_HM3`, returning
    /// `(col_lower[0], col_upper[0])` — the outgoing-storage column is always
    /// system index `0`.
    fn run_storage_fill(fixtures: &mut Fixtures, stage_id: i32) -> (f64, f64) {
        fixtures
            .bounds
            .hydro_bounds_mut(0, STAGE_IDX)
            .min_storage_hm3 = MIN_STORAGE_HM3;
        fixtures
            .bounds
            .hydro_bounds_mut(0, STAGE_IDX)
            .max_storage_hm3 = MAX_STORAGE_HM3;
        let stage_index = usize::try_from(stage_id).expect("test stage ids are non-negative");
        let stage = two_block_stage(stage_index, [372.0, 372.0]);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let (mut col_lower, mut col_upper, mut objective) = fresh_bufs(layout.num_cols);
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        super::fill_storage_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);
        (col_lower[0], col_upper[0])
    }

    /// A filling hydro relaxes its storage FLOOR to `0` in `PreFilling` and
    /// `Filling` (the reservoir may sit below dead volume while filling); the
    /// upper bound stays `max_storage` in every phase. The forbidden alternative —
    /// keeping the hard `min_storage` floor — would make the LP infeasible whenever
    /// the filling reservoir is below dead volume.
    #[test]
    fn filling_hydro_storage_floor_relaxed_in_prefilling_and_filling() {
        for stage_id in [PREFILLING_ID, FILLING_ID] {
            let mut fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
            let (lower, upper) = run_storage_fill(&mut fixtures, stage_id);
            assert_eq!(lower, 0.0, "storage col_lower relaxed to 0 at {stage_id}");
            assert_eq!(
                upper, MAX_STORAGE_HM3,
                "storage col_upper stays max_storage at {stage_id}"
            );
        }
    }

    /// A filling hydro relaxes its storage FLOOR to `0` in `Operating` too — the
    /// soft `σ^{v-}` operating-floor row supplies the economic floor so a hydro that
    /// finished filling short can recover without infeasibility. Combined with the
    /// `PreFilling`/`Filling` relax above, a filling hydro has `col_lower = 0` in ALL
    /// phases. The upper bound stays `max_storage`. The forbidden alternative —
    /// keeping the hard `min_storage` floor in Operating — would make the LP
    /// infeasible the moment a deficient-start reservoir sits below dead volume.
    #[test]
    fn filling_hydro_storage_floor_relaxed_in_operating() {
        let mut fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
        let (lower, upper) = run_storage_fill(&mut fixtures, OPERATING_ID);
        assert_eq!(lower, 0.0, "storage col_lower relaxed to 0 in Operating");
        assert_eq!(upper, MAX_STORAGE_HM3, "storage col_upper in Operating");
    }

    /// A non-filling hydro keeps the hard `min_storage` floor at EVERY stage id
    /// (the parity-neutrality contract — the floor relax never fires). The
    /// forbidden alternative — relaxing the floor system-wide — would silently make
    /// dead volume soft for every reservoir.
    #[test]
    fn non_filling_hydro_storage_floor_hard_at_every_stage() {
        for stage_id in [PREFILLING_ID, FILLING_ID, OPERATING_ID] {
            let mut fixtures = Fixtures::new(None, None, false);
            let (lower, upper) = run_storage_fill(&mut fixtures, stage_id);
            assert_eq!(
                lower, MIN_STORAGE_HM3,
                "non-filling storage col_lower hard at {stage_id}"
            );
            assert_eq!(
                upper, MAX_STORAGE_HM3,
                "non-filling storage col_upper at {stage_id}"
            );
        }
    }

    /// A commissioning-dormant non-filling hydro relaxes its storage FLOOR to `0`
    /// while `PreFilling` (the frozen-identity row pins `v_h` to the inert IC
    /// storage, which a hard `min_storage` floor would reject), then RESTORES the
    /// hard `min_storage` floor from `entry` onward (`Operating` — a normal plant
    /// with no soft operating-floor row, unlike a filling hydro). The forbidden
    /// alternative — keeping the relax at `Operating` — would silently make this
    /// plant's dead volume soft once it commissions.
    #[test]
    fn dormant_non_filling_hydro_storage_floor_relaxed_then_restored() {
        for stage_id in [PREFILLING_ID, FILLING_ID] {
            let mut fixtures = Fixtures::new(None, Some(ENTRY_STAGE_ID), false);
            let (lower, upper) = run_storage_fill(&mut fixtures, stage_id);
            assert_eq!(
                lower, 0.0,
                "dormant storage col_lower relaxed to 0 at {stage_id}"
            );
            assert_eq!(upper, MAX_STORAGE_HM3, "storage col_upper at {stage_id}");
        }
        let mut fixtures = Fixtures::new(None, Some(ENTRY_STAGE_ID), false);
        let (lower, upper) = run_storage_fill(&mut fixtures, OPERATING_ID);
        assert_eq!(
            lower, MIN_STORAGE_HM3,
            "storage col_lower restored to hard min_storage at Operating"
        );
        assert_eq!(upper, MAX_STORAGE_HM3, "storage col_upper at Operating");
    }

    // ── σ_fill terminal-target column ────────────────────────────────────────

    /// A representative non-default `filling_target_violation_cost` so the
    /// objective coefficient is observable against the `0.0` default.
    const FILLING_TARGET_COST: f64 = 50_000.0;

    /// Run `fill_filling_target_columns` against the fixture at `stage_id` with the
    /// resolved `filling_target_violation_cost` overridden to
    /// `FILLING_TARGET_COST`, returning the whole column buffers and the `σ_fill`
    /// column start (which equals `num_cols` when the block is empty).
    fn run_filling_target_fill(
        fixtures: &mut Fixtures,
        stage_id: i32,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, usize, usize) {
        fixtures
            .penalties
            .hydro_penalties_mut(0, STAGE_IDX)
            .filling_target_violation_cost = FILLING_TARGET_COST;
        let stage_index = usize::try_from(stage_id).expect("test stage ids are non-negative");
        let stage = two_block_stage(stage_index, [372.0, 372.0]);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let (mut col_lower, mut col_upper, mut objective) = fresh_bufs(layout.num_cols);
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        super::fill_filling_target_columns(&ctx, STAGE_IDX, &layout, &mut bufs);
        let n_targets = layout.filling.filling_target_hydro_indices.len();
        let col_start = layout.filling.col_filling_target_start;
        (
            col_lower,
            col_upper,
            objective,
            col_start,
            n_targets,
            layout.num_cols,
        )
    }

    /// At a Filling stage (every `id` in `[start, entry)`) a filling hydro gets
    /// exactly ONE `σ_fill` column: `col_lower = 0`, `col_upper = +∞`, and the
    /// objective coefficient is the RESOLVED `filling_target_violation_cost`
    /// UNSCALED — NOT multiplied by stage hours. The hours-multiplication
    /// alternative (copied from the flow/power-rate slacks) would be a $·h/hm³ units
    /// error on this storage-volume ($/hm³) penalty. Exercises BOTH Filling stages
    /// (ids 2 and 3) to pin per-stage membership, not the v1 terminal-only rule.
    #[test]
    fn filling_target_column_emitted_at_every_filling_stage() {
        // start = 2, entry = 4 ⇒ Filling stages are ids 2 and 3.
        for stage_id in [START_STAGE_ID, ENTRY_STAGE_ID - 1] {
            let mut fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
            let (col_lower, col_upper, objective, col_start, n_targets, _num_cols) =
                run_filling_target_fill(&mut fixtures, stage_id);
            assert_eq!(
                n_targets, 1,
                "exactly one σ_fill column at Filling id {stage_id}"
            );
            let col = col_start;
            assert_eq!(col_lower[col], 0.0, "σ_fill col_lower = 0 at id {stage_id}");
            assert_eq!(
                col_upper[col],
                f64::INFINITY,
                "σ_fill col_upper = +∞ at id {stage_id}"
            );
            // Cost is UNSCALED here (the global /COST_SCALE_FACTOR pass runs later in
            // build_single_stage_template) and carries NO hours factor.
            assert_eq!(
                objective[col], FILLING_TARGET_COST,
                "σ_fill objective = filling_target_violation_cost (unscaled, no hours) at id {stage_id}"
            );
        }
    }

    /// No `σ_fill` column is emitted off the Filling phase — `PreFilling` (id 1) or
    /// `Operating` (id 4). Per-stage Filling membership, NOT the v1 terminal-only
    /// rule.
    ///
    /// At `PreFilling` the `σ_fill` block being empty means its cursor coincides
    /// with `num_cols` (`σ_fill` is the last occupied family there, since `σ^{v-}`
    /// is also empty off `Operating`). At `Operating` the `σ^{v-}` block
    /// legitimately occupies one column AFTER the (empty) `σ_fill` block, so the
    /// faithful empty-`σ_fill` check is the zero block width (`n_targets == 0`), not
    /// the `== num_cols` coincidence — which `σ^{v-}` breaks at `Operating`.
    #[test]
    fn filling_target_column_absent_off_filling_phase() {
        for stage_id in [PREFILLING_ID, OPERATING_ID] {
            let mut fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
            let (_, _, _, col_start, n_targets, num_cols) =
                run_filling_target_fill(&mut fixtures, stage_id);
            assert_eq!(n_targets, 0, "no σ_fill column at id {stage_id}");
            if stage_id == OPERATING_ID {
                // σ^{v-} occupies exactly one column beyond the empty σ_fill block.
                assert_eq!(
                    col_start + 1,
                    num_cols,
                    "Operating: σ^{{v-}} column sits after the empty σ_fill block"
                );
            } else {
                assert_eq!(
                    col_start, num_cols,
                    "empty σ_fill block: col start coincides with num_cols at id {stage_id}"
                );
            }
        }
    }

    /// A non-filling hydro never gets a `σ_fill` column at any stage id
    /// (parity-neutral): the index list is empty and the column buffers are
    /// untouched by the fill.
    #[test]
    fn non_filling_hydro_no_filling_target_column() {
        for stage_id in [PREFILLING_ID, FILLING_ID, OPERATING_ID] {
            let mut fixtures = Fixtures::new(None, None, false);
            let (col_lower, col_upper, objective, col_start, n_targets, num_cols) =
                run_filling_target_fill(&mut fixtures, stage_id);
            assert_eq!(
                n_targets, 0,
                "non-filling: no σ_fill column at id {stage_id}"
            );
            assert_eq!(col_start, num_cols, "empty σ_fill block at id {stage_id}");
            // The fill wrote nothing: objective is all-zero, bounds are the fresh
            // `[0, +∞]` defaults (no column carries the penalty).
            assert!(
                objective.iter().all(|&c| c == 0.0),
                "non-filling: no objective entry written by σ_fill fill"
            );
            assert!(col_lower.iter().all(|&l| l == 0.0));
            assert!(col_upper.iter().all(|&u| u == f64::INFINITY));
        }
    }

    // ── σ^{v-} operating-floor column ────────────────────────────────────────

    /// A representative non-default `storage_violation_below_cost` so the objective
    /// coefficient is observable against the `0.0` default, and DISTINCT from
    /// `FILLING_TARGET_COST` so a test that conflates the two costs fails.
    const STORAGE_BELOW_COST: f64 = 12_345.0;

    /// Run `fill_filled_min_storage_floor_columns` against the fixture at `stage_id` with the
    /// resolved `storage_violation_below_cost` overridden to `STORAGE_BELOW_COST`,
    /// returning the column buffers and the `σ^{v-}` column start (which equals
    /// `num_cols` when the block is empty), the column count, and `num_cols`.
    fn run_filled_min_storage_floor_fill(
        fixtures: &mut Fixtures,
        stage_id: i32,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, usize, usize) {
        fixtures
            .penalties
            .hydro_penalties_mut(0, STAGE_IDX)
            .storage_violation_below_cost = STORAGE_BELOW_COST;
        let stage_index = usize::try_from(stage_id).expect("test stage ids are non-negative");
        let stage = two_block_stage(stage_index, [372.0, 372.0]);
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let (mut col_lower, mut col_upper, mut objective) = fresh_bufs(layout.num_cols);
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        super::fill_filled_min_storage_floor_columns(&ctx, STAGE_IDX, &layout, &mut bufs);
        let n_floors = layout.filling.filled_min_storage_floor_hydro_indices.len();
        let col_start = layout.filling.col_filled_min_storage_floor_start;
        (
            col_lower,
            col_upper,
            objective,
            col_start,
            n_floors,
            layout.num_cols,
        )
    }

    /// In `Operating` (`id == entry == 4`) a filling hydro gets exactly ONE
    /// `σ^{v-}` column: `col_lower = 0`, `col_upper = +∞`, and the objective
    /// coefficient is the RESOLVED `storage_violation_below_cost` UNSCALED — NOT
    /// multiplied by stage hours (the hours-multiplication alternative copied from
    /// the flow/power-rate slacks would be a $·h/hm³ units error on this
    /// storage-volume ($/hm³) penalty, identical to the `σ_fill` convention).
    #[test]
    fn filled_min_storage_floor_column_emitted_in_operating() {
        let mut fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
        let (col_lower, col_upper, objective, col_start, n_floors, _num_cols) =
            run_filled_min_storage_floor_fill(&mut fixtures, OPERATING_ID);
        assert_eq!(n_floors, 1, "exactly one σ^{{v-}} column in Operating");
        let col = col_start;
        assert_eq!(col_lower[col], 0.0, "σ^{{v-}} col_lower = 0");
        assert_eq!(col_upper[col], f64::INFINITY, "σ^{{v-}} col_upper = +∞");
        // Cost is UNSCALED here (the global /COST_SCALE_FACTOR pass runs later in
        // build_single_stage_template) and carries NO hours factor.
        assert_eq!(
            objective[col], STORAGE_BELOW_COST,
            "σ^{{v-}} objective = storage_violation_below_cost (unscaled, no hours)"
        );
    }

    /// No `σ^{v-}` column is emitted at any non-operating stage of a filling hydro —
    /// `PreFilling` or `Filling`. `Operating`-only (the complement of `σ_fill`'s
    /// every-Filling-stage scope), so the `σ^{v-}` family never collides with `σ_fill`.
    #[test]
    fn filled_min_storage_floor_column_absent_in_prefilling_and_filling() {
        for stage_id in [PREFILLING_ID, FILLING_ID] {
            let mut fixtures = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
            let (_, _, _, col_start, n_floors, num_cols) =
                run_filled_min_storage_floor_fill(&mut fixtures, stage_id);
            assert_eq!(n_floors, 0, "no σ^{{v-}} column at id {stage_id}");
            assert_eq!(
                col_start, num_cols,
                "empty σ^{{v-}} block: col start coincides with num_cols at id {stage_id}"
            );
        }
    }

    /// A non-filling hydro never gets a `σ^{v-}` column at any stage id
    /// (parity-neutral): the index list is empty and the column buffers are
    /// untouched by the fill. The forbidden GLOBAL soft floor — matching every
    /// Operating hydro regardless of `filling` — would softly floor every reservoir.
    #[test]
    fn non_filling_hydro_no_filled_min_storage_floor_column() {
        for stage_id in [PREFILLING_ID, FILLING_ID, OPERATING_ID] {
            let mut fixtures = Fixtures::new(None, None, false);
            let (col_lower, col_upper, objective, col_start, n_floors, num_cols) =
                run_filled_min_storage_floor_fill(&mut fixtures, stage_id);
            assert_eq!(
                n_floors, 0,
                "non-filling: no σ^{{v-}} column at id {stage_id}"
            );
            assert_eq!(col_start, num_cols, "empty σ^{{v-}} block at id {stage_id}");
            assert!(
                objective.iter().all(|&c| c == 0.0),
                "non-filling: no objective entry written by σ^{{v-}} fill"
            );
            assert!(col_lower.iter().all(|&l| l == 0.0));
            assert!(col_upper.iter().all(|&u| u == f64::INFINITY));
        }
    }

    /// The `σ^{v-}` (`Operating`) and `σ_fill` (`Filling`) families are MUTUALLY
    /// EXCLUSIVE per stage: at the LAST Filling stage (`entry − 1`, the boundary
    /// witness) a filling hydro has `σ_fill` but NOT `σ^{v-}`; at the first
    /// `Operating` stage (`entry`) it has `σ^{v-}` but NOT `σ_fill`. Two separate
    /// columns, two non-overlapping stage scopes (Filling vs Operating) — never
    /// conflated.
    #[test]
    fn filled_min_storage_floor_and_filling_target_are_mutually_exclusive_by_stage() {
        // Last Filling stage (entry − 1): σ_fill present, σ^{v-} absent.
        let mut f_terminal = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
        let (_, _, _, _, n_targets_terminal, _) =
            run_filling_target_fill(&mut f_terminal, ENTRY_STAGE_ID - 1);
        let mut f_terminal2 = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
        let (_, _, _, _, n_floors_terminal, _) =
            run_filled_min_storage_floor_fill(&mut f_terminal2, ENTRY_STAGE_ID - 1);
        assert_eq!(n_targets_terminal, 1, "σ_fill present at terminal stage");
        assert_eq!(n_floors_terminal, 0, "σ^{{v-}} absent at terminal stage");

        // Operating stage (entry): σ^{v-} present, σ_fill absent.
        let mut f_op = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
        let (_, _, _, _, n_targets_op, _) = run_filling_target_fill(&mut f_op, OPERATING_ID);
        let mut f_op2 = Fixtures::new(Some(filling_config()), Some(ENTRY_STAGE_ID), false);
        let (_, _, _, _, n_floors_op, _) =
            run_filled_min_storage_floor_fill(&mut f_op2, OPERATING_ID);
        assert_eq!(n_targets_op, 0, "σ_fill absent in Operating");
        assert_eq!(n_floors_op, 1, "σ^{{v-}} present in Operating");
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod anticipated_objective_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::thermal::AnticipatedConfig;
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, CascadeTopology, ContractStageBounds, EntityId,
        HydroStageBounds, LineStageBounds, PumpingStageBounds, ResolvedBounds,
        ResolvedExchangeFactors, ResolvedGenericConstraintBounds, ResolvedLoadFactors,
        ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties, Thermal, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::{state_layout_for, two_block_stage};
    use super::{StageLayout, TemplateBuildCtx, fill_stage_columns};
    use crate::indexer::ThermalSys;

    const N_STAGES: usize = 6;
    const K_MAX: usize = 1;
    const DELIVERY_COST_PER_MWH: f64 = 30.0;
    const STD_COST_PER_MWH: f64 = 40.0;
    const MAX_GEN_MW: f64 = 100.0;
    const STAGE_IDX: usize = 0;
    const DELIVERY_STAGE: usize = STAGE_IDX + 1; // K_0 = 1.

    /// Owns the borrow targets for a one-anticipated-thermal `TemplateBuildCtx`.
    ///
    /// Thermal 0 is anticipated (`K=1`), thermal 1 is a standard thermal. Both
    /// carry a non-zero resolved `cost_per_mwh` so the R3 skip and the R1 NPV
    /// objective are both observable in the assertions.
    struct AntObjFixtures {
        par_lp: PrecomputedPar,
        thermals: Vec<Thermal>,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_exchange_factors: ResolvedExchangeFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
    }

    impl AntObjFixtures {
        fn new() -> Self {
            let thermals = vec![
                Thermal {
                    id: EntityId(1),
                    name: "T_ant".to_string(),
                    operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    bus_id: EntityId(1),
                    min_generation_mw: 0.0,
                    max_generation_mw: MAX_GEN_MW,
                    cost_per_mwh: DELIVERY_COST_PER_MWH,
                    // lead_stages == K_MAX; the entity field is u32 while K_MAX
                    // is the usize layout dimension, so write the value directly.
                    anticipated_config: Some(AnticipatedConfig::LeadStages(1)),
                    entry_stage_id: None,
                    exit_stage_id: None,
                },
                Thermal {
                    id: EntityId(2),
                    name: "T_std".to_string(),
                    operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                    bus_id: EntityId(1),
                    min_generation_mw: 0.0,
                    max_generation_mw: MAX_GEN_MW,
                    cost_per_mwh: STD_COST_PER_MWH,
                    anticipated_config: None,
                    entry_stage_id: None,
                    exit_stage_id: None,
                },
            ];
            let mut bounds = bounds_two_thermals();
            for stage in 0..N_STAGES {
                let ant = bounds.thermal_bounds_mut(0, stage);
                ant.cost_per_mwh = DELIVERY_COST_PER_MWH;
                ant.max_generation_mw = MAX_GEN_MW;
                let std = bounds.thermal_bounds_mut(1, stage);
                std.cost_per_mwh = STD_COST_PER_MWH;
                std.max_generation_mw = MAX_GEN_MW;
            }
            Self {
                par_lp: PrecomputedPar::default(),
                thermals,
                cascade: CascadeTopology::build(&[]),
                bounds,
                penalties: ResolvedPenalties::empty(),
                production_models: ProductionModelSet::new(vec![], 0, 1),
                evaporation_models: EvaporationModelSet::new(vec![]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
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
            TemplateBuildCtx {
                hydros: &[],
                thermals: &self.thermals,
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
                hydro_pos: BTreeMap::new(),
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
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                n_hydros: 0,
                n_thermals: 2,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 1,
                k_max: K_MAX,
                anticipated_lead_stages: vec![K_MAX],
                anticipated_thermal_indices: vec![ThermalSys::new(0)],
                // Windowless single plant: the decision gate reduces to the
                // strict horizon clause. `study_stage_ids` lists the N_STAGES
                // study-stage ids so the in-range delivery lookup is safe.
                anticipated_windows: vec![(None, None)],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: (0..N_STAGES as i32).collect(),
                has_penalty: false,
                cumulative_discount_factors: vec![1.0, 0.9, 0.81, 0.729, 0.6561, 0.59049],
                total_hours_per_stage: vec![744.0; N_STAGES],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// `ResolvedBounds` table for two thermals across `N_STAGES` stages.
    fn bounds_two_thermals() -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 2,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages: N_STAGES,
                k_max: K_MAX,
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
                    filling_min_rate_m3s: 0.0,
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

    /// After `fill_stage_columns`, the anticipated thermal's per-block delivery
    /// objective is `0.0` (R3: `fill_thermal_columns` skips the objective write),
    /// while the standard thermal is priced normally; and the anticipated
    /// decision column carries the NPV-discounted commitment cost (R1: the merged
    /// `fill_anticipated_columns` writes `cost * hours * cumulative_discount`).
    #[test]
    fn anticipated_objective_skip_and_npv_after_fill_stage_columns() {
        let fixtures = AntObjFixtures::new();
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(STAGE_IDX, [372.0, 372.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);

        let (_col_lower, col_upper, objective) =
            fill_stage_columns(&ctx, &stage, STAGE_IDX, &layout);

        let n_blks = layout.n_blks;
        // R3: anticipated thermal (t_idx 0) objective stays at the 0.0 default;
        // its per-block bounds are still written by fill_thermal_columns.
        for blk in 0..n_blks {
            let col = layout.equipment.thermal.start + blk;
            assert_eq!(
                objective[col], 0.0,
                "anticipated thermal objective must be 0.0 at col {col}",
            );
            assert_eq!(
                col_upper[col], MAX_GEN_MW,
                "anticipated thermal per-block bounds must still be set at col {col}",
            );
        }
        // R3 control: standard thermal (t_idx 1) is priced as cost * block_hours.
        for blk in 0..n_blks {
            let col = layout.equipment.thermal.start + n_blks + blk;
            let expected = STD_COST_PER_MWH * stage.blocks[blk].duration_hours;
            assert_eq!(
                objective[col], expected,
                "standard thermal objective must be priced at col {col}",
            );
        }
        // R1: anticipated decision column carries the NPV commitment cost
        // cost_per_mwh(delivery) * total_hours[delivery] * cumulative_discount[delivery].
        let decision_col = layout.anticipated.col_anticipated_decision_start;
        let expected_npv = DELIVERY_COST_PER_MWH
            * ctx.total_hours_per_stage[DELIVERY_STAGE]
            * ctx.cumulative_discount_factors[DELIVERY_STAGE];
        assert_eq!(
            objective[decision_col], expected_npv,
            "anticipated decision objective must equal the NPV commitment cost",
        );
        // The active plant's newest ring slot is open (active), confirming the
        // merged fill ran the active branch. K_MAX == 1 here, so the newest
        // slot is the ring's own start (no per-plant offset needed).
        let state_out_col = layout.anticipated.col_anticipated_slots_out_start;
        assert_eq!(col_upper[state_out_col], f64::INFINITY);
    }

    /// Borrow-target owner for a one-anticipated-plant delivery-anchoring
    /// preservation `TemplateBuildCtx`. `per_stage[s] == (min_gen, max_gen,
    /// cost)` is the plant's stage-`s` `thermal_bounds`; the discount factor is
    /// stage-varying (`0.9^s`) too. Both are deliberately stage-varying so a
    /// DECISION-anchored read (the forbidden alternative) yields a provably
    /// different column than the shipped DELIVERY-anchored read.
    struct DeliveryAnchoredFixtures {
        thermals: Vec<Thermal>,
        cascade: CascadeTopology,
        bounds: ResolvedBounds,
        par_lp: PrecomputedPar,
        penalties: ResolvedPenalties,
        production_models: ProductionModelSet,
        evaporation_models: EvaporationModelSet,
        resolved_generic_bounds: ResolvedGenericConstraintBounds,
        resolved_load_factors: ResolvedLoadFactors,
        resolved_exchange_factors: ResolvedExchangeFactors,
        resolved_ncs_bounds: ResolvedNcsBounds,
        resolved_ncs_factors: ResolvedNcsFactors,
        resolved_parameters: ResolvedParameters,
        n_stages: usize,
        k_max: usize,
        discount: Vec<f64>,
        hours: Vec<f64>,
    }

    impl DeliveryAnchoredFixtures {
        fn new(
            n_stages: usize,
            k_max: usize,
            config: AnticipatedConfig,
            per_stage: &[(f64, f64, f64)],
        ) -> Self {
            let thermals = vec![Thermal {
                id: EntityId(1),
                name: "T_ant".to_string(),
                operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: 0.0,
                anticipated_config: Some(config),
                entry_stage_id: None,
                exit_stage_id: None,
            }];
            let mut bounds = ResolvedBounds::new(
                &BoundsCountsSpec {
                    n_hydros: 0,
                    n_thermals: 1,
                    n_lines: 0,
                    n_pumping: 0,
                    n_contracts: 0,
                    n_stages,
                    k_max,
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
                        filling_min_rate_m3s: 0.0,
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
            for (stage, &(min_g, max_g, cost)) in per_stage.iter().enumerate() {
                let tb = bounds.thermal_bounds_mut(0, stage);
                tb.min_generation_mw = min_g;
                tb.max_generation_mw = max_g;
                tb.cost_per_mwh = cost;
            }
            let mut discount = Vec::with_capacity(n_stages);
            let mut d = 1.0_f64;
            for _ in 0..n_stages {
                discount.push(d);
                d *= 0.9;
            }
            Self {
                thermals,
                cascade: CascadeTopology::build(&[]),
                bounds,
                par_lp: PrecomputedPar::default(),
                penalties: ResolvedPenalties::empty(),
                production_models: ProductionModelSet::new(vec![], 0, 1),
                evaporation_models: EvaporationModelSet::new(vec![]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_exchange_factors: ResolvedExchangeFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                },
                n_stages,
                k_max,
                discount,
                hours: vec![744.0; n_stages],
            }
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            TemplateBuildCtx {
                hydros: &[],
                thermals: &self.thermals,
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
                hydro_pos: BTreeMap::new(),
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
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                n_hydros: 0,
                n_thermals: 1,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 1,
                k_max: self.k_max,
                anticipated_lead_stages: vec![self.k_max],
                anticipated_thermal_indices: vec![ThermalSys::new(0)],
                anticipated_windows: vec![(None, None)],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: (0..self.n_stages as i32).collect(),
                has_penalty: false,
                cumulative_discount_factors: self.discount.clone(),
                total_hours_per_stage: self.hours.clone(),
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// Delivery-anchoring preservation contract: the anticipated decision
    /// column is bounded/costed at ITS OWN delivery stage `m`, never the
    /// decision stage —
    /// `col_upper == thermal_bounds(m).max_generation_mw`,
    /// `col_lower == thermal_bounds(m).min_generation_mw`,
    /// `objective == cost(m) * hours(m) * discount(m)`. STAGE-VARYING delivery
    /// bounds/cost so a decision-anchored read gives a provably different
    /// value (constant-across-lead bounds would make the test vacuous).
    ///
    /// Load-bearing verified by MUTATION: changing the production read at
    /// `fill_anticipated_columns` from `thermal_bounds(thermal_idx,
    /// delivery_stage)` to `thermal_bounds(thermal_idx, stage_idx)` (the
    /// decision stage) fails the first assertion — `left: 55.0, right: 100.0`
    /// on `col_upper[decision]` — reintroducing the capacity-drop
    /// infeasibility this contract forbids.
    #[test]
    fn test_anticipated_decision_delivery_anchored_bounds() {
        // Decision stage 0, delivery stage 1. Stage 0's bounds/cost differ
        // from stage 1's, so a decision-anchored read (stage 0) is
        // distinguishable from the delivery-anchored read.
        const N_STAGES: usize = 6;
        const K_MAX: usize = 1;
        // (min_gen, max_gen, cost) per stage; only stage 1 (delivery) is
        // read by the fill, stage 0 (decision) is the discriminating decoy.
        let per_stage = [
            (5.0, 55.0, 15.0),
            (11.0, 100.0, 30.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ];
        let fx = DeliveryAnchoredFixtures::new(
            N_STAGES,
            K_MAX,
            AnticipatedConfig::LeadStages(1),
            &per_stage,
        );
        let ctx = fx.make_ctx();
        let stage = two_block_stage(0, [372.0, 372.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let (col_lower, col_upper, objective) = fill_stage_columns(&ctx, &stage, 0, &layout);
        let decision_col = layout.anticipated.col_anticipated_decision_start;

        let delivery = 1_usize;
        let (min_g, max_g, cost) = per_stage[delivery];
        assert_eq!(
            col_upper[decision_col], max_g,
            "decision column must bound at its OWN delivery stage {delivery}'s \
             max_generation_mw, not the decision stage's",
        );
        assert_eq!(
            col_lower[decision_col], min_g,
            "decision column must bound at its OWN delivery stage {delivery}'s \
             min_generation_mw, not the decision stage's",
        );
        let expected_obj =
            cost * ctx.total_hours_per_stage[delivery] * ctx.cumulative_discount_factors[delivery];
        assert_eq!(
            objective[decision_col], expected_obj,
            "decision objective must be priced at its OWN delivery stage \
             {delivery}'s cost/hours/discount",
        );
        // The decision stage's own bounds (stage 0) must be strictly
        // different, or the anchoring proof is vacuous.
        let (dec_min, dec_max, _) = per_stage[0];
        assert_ne!(max_g, dec_max, "delivery and decision max_gen must differ");
        assert_ne!(min_g, dec_min, "delivery and decision min_gen must differ");
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod block_family_slack_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, BusStagePenalties, CascadeTopology, ContractStageBounds,
        EntityId, Hydro, HydroStageBounds, HydroStagePenalties, LineStageBounds,
        LineStagePenalties, NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults,
        PumpingStageBounds, ResolvedBounds, ResolvedExchangeFactors,
        ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds,
        ResolvedNcsFactors, ResolvedPenalties, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, ProductionModelSet, ResolvedProductionModel,
    };
    use crate::indexer::{BlockIdx, HydroSys};
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::{state_layout_for, two_block_stage, zero_hydro_penalties};
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
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
            travel_time_hours: None,
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
            filling_min_rate_m3s: 0.0,
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
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
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
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                n_hydros: N_HYDROS,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: vec![],
                has_penalty: false,
                cumulative_discount_factors: vec![1.0],
                total_hours_per_stage: vec![BLOCK_HOURS[0] + BLOCK_HOURS[1]],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// One family's expected contract: its name, the activation predicate over a
    /// `HydroSpec`, the `StageLayout` column accessor, and the expected cost field.
    struct FamilyCheck<'b> {
        name: &'static str,
        predicate: fn(&HydroSpec) -> bool,
        accessor: fn(&StageLayout<'b>, HydroSys, BlockIdx) -> usize,
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
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);

        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_operational_slack_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);

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
                    let col = (family.accessor)(&layout, HydroSys::new(h_idx), BlockIdx::new(blk));
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

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod evaporation_slack_objective_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::hydro::HydroGenerationModel;
    use cobre_core::{
        Block, BlockMode, BoundsCountsSpec, BoundsDefaults, BusStagePenalties, CascadeTopology,
        ContractStageBounds, EntityId, Hydro, HydroStageBounds, HydroStagePenalties,
        LineStageBounds, LineStagePenalties, NcsStagePenalties, NoiseMethod, PenaltiesCountsSpec,
        PenaltiesDefaults, PumpingStageBounds, ResolvedBounds, ResolvedExchangeFactors,
        ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds,
        ResolvedNcsFactors, ResolvedPenalties, ScenarioSourceConfig, Stage, StageRiskConfig,
        StageStateConfig, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{
        EvaporationModel, EvaporationModelSet, LinearizedEvaporation, ProductionModelSet,
        ResolvedProductionModel,
    };
    use crate::indexer::{BlockIdx, EvapLocal};
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::{state_layout_for, zero_hydro_penalties};
    use super::{ColumnBufs, StageLayout, TemplateBuildCtx, fill_evaporation_columns};

    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;
    const NEG_COST: f64 = 3.0;
    const POS_COST: f64 = 7.0;

    /// One Operating hydro carrying a `Linearized` evaporation model, so
    /// `identify_evap_hydros` reserves the `EVAP_COLS_PER_HYDRO` triple per block.
    fn evaporating_hydro() -> Hydro {
        Hydro {
            id: EntityId(1),
            name: "H1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: None,
            travel_time_hours: None,
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
                    max_storage_hm3: 100.0,
                    min_turbined_m3s: 0.0,
                    max_turbined_m3s: 50.0,
                    min_outflow_m3s: 0.0,
                    max_outflow_m3s: None,
                    min_generation_mw: 0.0,
                    max_generation_mw: 45.0,
                    max_diversion_m3s: None,
                    filling_min_rate_m3s: 0.0,
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

    /// One hydro's penalties with distinct nonzero evaporation-violation costs so a
    /// pos/neg cross-wire and a missing per-block weighting are both observable.
    fn penalties_one_hydro() -> ResolvedPenalties {
        let mut penalties = ResolvedPenalties::new(
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
        );
        let hp = penalties.hydro_penalties_mut(0, STAGE_IDX);
        hp.evaporation_violation_neg_cost = NEG_COST;
        hp.evaporation_violation_pos_cost = POS_COST;
        penalties
    }

    /// A `Stage` with `block_durations.len()` blocks under `block_mode`. The
    /// durations differ per block so a per-block divisor confusion in the code under
    /// test cannot be masked by equal blocks.
    fn stage_with_blocks(block_mode: BlockMode, block_durations: &[f64]) -> Stage {
        Stage {
            index: STAGE_IDX,
            id: STAGE_IDX as i32,
            start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: chrono::NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: block_durations
                .iter()
                .enumerate()
                .map(|(index, &duration_hours)| Block {
                    index,
                    name: format!("BLK{index}"),
                    duration_hours,
                })
                .collect(),
            block_mode,
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

    /// Owns the borrow targets for a one-evaporating-hydro `TemplateBuildCtx`.
    struct EvapFixtures {
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

    impl EvapFixtures {
        fn new() -> Self {
            let hydros = vec![evaporating_hydro()];
            let cascade = CascadeTopology::build(&hydros);
            let evaporation_models = EvaporationModelSet::new(vec![EvaporationModel::Linearized {
                coefficients: vec![
                    LinearizedEvaporation {
                        intercept_m3s: 1.0,
                        volume_slope_m3s_per_hm3: 0.0,
                    };
                    N_STAGES
                ],
                reference_volumes_hm3: vec![50.0; N_STAGES],
            }]);
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
                evaporation_models,
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
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
                contracts: &[],
                contract_pos: BTreeMap::new(),
                n_contract_import: 0,
                n_contract_export: 0,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                n_hydros: 1,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: vec![],
                has_penalty: false,
                cumulative_discount_factors: vec![1.0],
                total_hours_per_stage: vec![744.0],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// Per-block `f_evap_plus`/`f_evap_minus` objectives plus the layout's `n_blks`.
    struct EvapFill {
        f_plus: Vec<f64>,
        f_minus: Vec<f64>,
        n_blks: usize,
    }

    fn run_fill(fixtures: &EvapFixtures, stage: &Stage) -> EvapFill {
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, stage, STAGE_IDX);
        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_evaporation_columns(&ctx, stage, STAGE_IDX, &layout, &mut bufs);
        let f_plus = (0..layout.n_blks)
            .map(|blk| objective[layout.evap_f_plus_col(EvapLocal::new(0), BlockIdx::new(blk))])
            .collect();
        let f_minus = (0..layout.n_blks)
            .map(|blk| objective[layout.evap_f_minus_col(EvapLocal::new(0), BlockIdx::new(blk))])
            .collect();
        EvapFill {
            f_plus,
            f_minus,
            n_blks: layout.n_blks,
        }
    }

    /// In chronological K ≥ 2 each block's evaporation-violation slack objective is
    /// the directional cost times THAT block's `duration_hours` — not the stage-total
    /// hours on every block (the pre-fix inflation) — and the per-block sum telescopes
    /// to `cost * total_stage_hours` (the single-slack parallel total).
    #[test]
    fn chronological_evap_slack_objective_is_block_weighted() {
        let block_durations = [300.0, 444.0, 148.0];
        let total_hours: f64 = block_durations.iter().sum();
        let fixtures = EvapFixtures::new();
        let chrono = run_fill(
            &fixtures,
            &stage_with_blocks(BlockMode::Chronological, &block_durations),
        );

        assert_eq!(
            chrono.n_blks,
            block_durations.len(),
            "layout must reserve one evap triple per block"
        );
        for (blk, &hours) in block_durations.iter().enumerate() {
            assert_eq!(
                chrono.f_plus[blk],
                NEG_COST * hours,
                "blk {blk}: f_evap_plus objective must be neg_cost * this block's hours"
            );
            assert_eq!(
                chrono.f_minus[blk],
                POS_COST * hours,
                "blk {blk}: f_evap_minus objective must be pos_cost * this block's hours"
            );
        }
        let plus_sum: f64 = chrono.f_plus.iter().sum();
        let minus_sum: f64 = chrono.f_minus.iter().sum();
        assert_eq!(
            plus_sum,
            NEG_COST * total_hours,
            "Σ f_evap_plus over blocks must telescope to neg_cost * total_stage_hours"
        );
        assert_eq!(
            minus_sum,
            POS_COST * total_hours,
            "Σ f_evap_minus over blocks must telescope to pos_cost * total_stage_hours"
        );
    }

    /// Parallel (`n_blks == 1`): the single evaporation slack objective is
    /// `cost * total_stage_hours` (`blocks[0].duration_hours == total_stage_hours`),
    /// unchanged by the per-block weighting.
    #[test]
    fn parallel_evap_slack_objective_equals_total_stage_hours() {
        let total_hours = 744.0;
        let fixtures = EvapFixtures::new();
        let parallel = run_fill(
            &fixtures,
            &stage_with_blocks(BlockMode::Parallel, &[total_hours]),
        );

        assert_eq!(parallel.n_blks, 1, "parallel mode reserves one block");
        assert_eq!(
            parallel.f_plus[0],
            NEG_COST * total_hours,
            "parallel f_evap_plus objective must equal neg_cost * total_stage_hours"
        );
        assert_eq!(
            parallel.f_minus[0],
            POS_COST * total_hours,
            "parallel f_evap_minus objective must equal pos_cost * total_stage_hours"
        );
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod contract_column_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::energy_contract::{ContractType, EnergyContract};
    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, CascadeTopology, ContractStageBounds, EntityId,
        HydroStageBounds, LineStageBounds, PumpingStageBounds, ResolvedBounds,
        ResolvedExchangeFactors, ResolvedGenericConstraintBounds, ResolvedLoadFactors,
        ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
    use crate::lead_time::AnticipatedResolution;
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::layout::ResolvedTables;
    use super::super::test_support::state_layout_for;
    use super::{ColumnBufs, StageLayout, TemplateBuildCtx, fill_contract_columns};

    const N_STAGES: usize = 1;
    const STAGE_IDX: usize = 0;
    const BLOCK_HOURS: f64 = 730.0;

    fn contract(
        id: i32,
        contract_type: ContractType,
        entry_stage_id: Option<i32>,
    ) -> EnergyContract {
        EnergyContract {
            id: EntityId(id),
            name: format!("C{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            contract_type,
            entry_stage_id,
            exit_stage_id: None,
            price_per_mwh: 0.0,
            min_mw: 0.0,
            max_mw: 0.0,
        }
    }

    fn bounds_with_contracts(n_contracts: usize) -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_pumping: 0,
                n_contracts,
                n_stages: N_STAGES,
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
                    filling_min_rate_m3s: 0.0,
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

    /// Owns the borrow targets for a contract-only `TemplateBuildCtx`.
    struct ContractFixtures {
        par_lp: PrecomputedPar,
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
        contracts: Vec<EnergyContract>,
    }

    impl ContractFixtures {
        fn new(contracts: Vec<EnergyContract>) -> Self {
            Self {
                par_lp: PrecomputedPar::default(),
                cascade: CascadeTopology::build(&[]),
                bounds: bounds_with_contracts(contracts.len()),
                penalties: ResolvedPenalties::empty(),
                production_models: ProductionModelSet::new(vec![], 0, 1),
                evaporation_models: EvaporationModelSet::new(vec![]),
                resolved_generic_bounds: ResolvedGenericConstraintBounds::empty(),
                resolved_load_factors: ResolvedLoadFactors::empty(),
                resolved_exchange_factors: ResolvedExchangeFactors::empty(),
                resolved_ncs_bounds: ResolvedNcsBounds::empty(),
                resolved_ncs_factors: ResolvedNcsFactors::empty(),
                resolved_parameters: ResolvedParameters {
                    per_param: vec![],
                    id_to_slot: vec![],
                },
                contracts,
            }
        }

        fn set_contract_bounds(&mut self, c_sys: usize, min_mw: f64, max_mw: f64, price: f64) {
            let cell = self.bounds.contract_bounds_mut(c_sys, STAGE_IDX);
            cell.min_mw = min_mw;
            cell.max_mw = max_mw;
            cell.price_per_mwh = price;
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            let n_contract_import = self
                .contracts
                .iter()
                .filter(|c| c.contract_type == ContractType::Import)
                .count();
            let n_contract_export = self
                .contracts
                .iter()
                .filter(|c| c.contract_type == ContractType::Export)
                .count();
            let contract_pos: BTreeMap<EntityId, usize> = self
                .contracts
                .iter()
                .enumerate()
                .map(|(i, c)| (c.id, i))
                .collect();
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
                hydro_pos: BTreeMap::new(),
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
                contracts: &self.contracts,
                contract_pos,
                n_contract_import,
                n_contract_export,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: AnticipatedResolution::default(),
                study_stage_ids: vec![],
                has_penalty: false,
                cumulative_discount_factors: vec![1.0; N_STAGES],
                total_hours_per_stage: vec![744.0; N_STAGES],
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// Single-block stage at `STAGE_IDX` with `id = 0` and `BLOCK_HOURS` duration.
    fn one_block_stage() -> cobre_core::Stage {
        use chrono::NaiveDate;
        use cobre_core::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };
        Stage {
            index: STAGE_IDX,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: BLOCK_HOURS,
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
        }
    }

    /// Run `fill_contract_columns` and return `(col_lower, col_upper, objective)`
    /// plus the two family-base offsets the assertions read.
    fn run_fill(fixtures: &ContractFixtures) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, usize) {
        let stage = one_block_stage();
        let ctx = fixtures.make_ctx();
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, STAGE_IDX);
        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };
        fill_contract_columns(&ctx, &stage, STAGE_IDX, &layout, &mut bufs);
        (
            col_lower,
            col_upper,
            objective,
            layout.equipment.col_contract_import_start,
            layout.equipment.col_contract_export_start,
        )
    }

    /// An active import contract carries its resolved `[min_mw, max_mw]` bounds and a
    /// `price * block_hours` objective with the stored (positive = cost) sign.
    #[test]
    fn import_contract_bounds_and_objective() {
        let mut fixtures = ContractFixtures::new(vec![contract(1, ContractType::Import, None)]);
        fixtures.set_contract_bounds(0, 10.0, 100.0, 200.0);

        let (col_lower, col_upper, objective, import_start, _) = run_fill(&fixtures);
        assert_eq!(col_lower[import_start], 10.0);
        assert_eq!(col_upper[import_start], 100.0);
        assert_eq!(objective[import_start], 200.0 * BLOCK_HOURS);
    }

    /// An active export contract keeps the stored negative price sign in the
    /// objective — the wiring must NOT negate it.
    #[test]
    fn export_contract_objective_keeps_negative_sign() {
        let mut fixtures = ContractFixtures::new(vec![contract(1, ContractType::Export, None)]);
        fixtures.set_contract_bounds(0, 0.0, 500.0, -150.0);

        let (_, _, objective, _, export_start) = run_fill(&fixtures);
        assert_eq!(objective[export_start], -150.0 * BLOCK_HOURS);
    }

    /// A contract whose entry window opens after the current stage is dormant: BOTH
    /// bounds are pinned to zero at every block (never `[min > 0, 0]`).
    #[test]
    fn dormant_contract_zero_pins_both_bounds() {
        let mut fixtures = ContractFixtures::new(vec![contract(1, ContractType::Import, Some(2))]);
        fixtures.set_contract_bounds(0, 25.0, 100.0, 200.0);

        let (col_lower, col_upper, _, import_start, _) = run_fill(&fixtures);
        assert_eq!(col_lower[import_start], 0.0);
        assert_eq!(col_upper[import_start], 0.0);
    }

    /// A take-or-pay floor (`min_mw > 0`) lands on `col_lower` as a hard lower bound.
    #[test]
    fn take_or_pay_floor_sets_col_lower() {
        let mut fixtures = ContractFixtures::new(vec![contract(1, ContractType::Import, None)]);
        fixtures.set_contract_bounds(0, 50.0, 100.0, 200.0);

        let (col_lower, _, _, import_start, _) = run_fill(&fixtures);
        assert_eq!(col_lower[import_start], 50.0);
    }
}
