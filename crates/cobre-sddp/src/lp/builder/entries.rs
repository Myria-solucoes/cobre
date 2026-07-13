use cobre_core::{BlockMode, CoefficientRef, ConstraintSense, ContractType, EntityId, Stage};

use crate::generic_constraints::resolve_variable_ref;
use crate::hydro_models::{EvaporationModel, ResolvedProductionModel};
use crate::indexer::{AnticipatedLocal, EvapLocal, HydroSys, LineSys, StateLayout};

use super::M3S_TO_HM3;
use super::delivery_ring::DeliveryRing;
use super::fpha_cursor::for_each_fpha_plane;
use super::layout::{StageLayout, TemplateBuildCtx};

/// The one dense anticipated-thermal ring (`n_lanes = n_anticipated`,
/// slot-major/plant-minor) every anticipated call site shares — the single owner
/// of its out/in block construction.
pub(super) fn anticipated_ring(layout: &StageLayout) -> DeliveryRing {
    let state = layout.state;
    DeliveryRing::new(
        state.anticipated_slots_out.clone(),
        state.anticipated_state.clone(),
        layout.n_anticipated,
        layout.k_max,
    )
}

/// Fishing coupling per genuinely-anticipated plant: summed per-block thermal
/// energy equals the slot-0 committed power scaled to `MWh` (`MW × block_hours`). A
/// `K = 0` self-delivery excludes the plant's row (`anticipated_fishing_row_pos`),
/// so its ordinary thermal generation carries no fishing coupling.
pub(super) fn fill_anticipated_fishing_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let grid = layout.block_grid();
    let ring = anticipated_ring(layout);
    let mut n_active = 0_usize;
    for local_idx in 0..ctx.n_anticipated {
        let Some(pos) = layout.anticipated.anticipated_fishing_row_pos[local_idx] else {
            continue;
        };
        let row = layout.anticipated.row_anticipated_fishing_start + pos;
        let thermal_idx = ctx.anticipated_thermal_indices[local_idx];
        let mut block_hours_total: f64 = 0.0;
        for blk in 0..n_blks {
            let col_gen = grid.flat(layout.equipment.thermal.start, thermal_idx.get(), blk);
            let block_hours = stage.blocks[blk].duration_hours;
            col_entries[col_gen].push((row, block_hours));
            block_hours_total += block_hours;
        }
        let col_state = ring.in_col(0, local_idx);
        col_entries[col_state].push((row, -block_hours_total));
        n_active += 1;
    }
    debug_assert_eq!(
        n_active, layout.anticipated.n_anticipated_fishing_rows,
        "fill_anticipated_fishing_entries: active count mismatch"
    );
}

/// Encode the anticipated ring's delivery-decision deposit row
/// `slot^out − decision_col = 0` for each plant with a genuine, active decision
/// this stage (`anticipated_decision_row_pos`). `slot = delivery_stage − stage_idx − 1`
/// — computed directly from the delivery stage, never a `depth`-derived boundary.
pub(super) fn fill_anticipated_state_out_def_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_stages = ctx.resolved.bounds.n_stages();
    let n_ant = ctx.n_anticipated;
    let row_start = layout.anticipated.row_anticipated_state_out_def_start;
    let ring = anticipated_ring(layout);
    let mut n_active: usize = 0;
    for local_idx in 0..n_ant {
        let point = layout
            .state
            .anticipated_resolution_for(AnticipatedLocal::new(local_idx), n_stages);
        let Some(delivery_stage) = point.genuine_decisions_at(stage_idx).next() else {
            continue;
        };
        let Some(pos) = layout.anticipated.anticipated_decision_row_pos[local_idx] else {
            continue;
        };
        let row = row_start + pos;
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
        let col_decision = layout.anticipated.col_anticipated_decision_start + local_idx;
        ring.emit_deposit(slot, local_idx, row, col_decision, col_entries);
        n_active += 1;
    }
    debug_assert_eq!(
        n_active, layout.anticipated.n_anticipated_state_out_def_rows,
        "fill_anticipated_state_out_def_entries: active count mismatch at stage {stage_idx}"
    );
}

/// Encode the anticipated ring's interior shift rows `slot_k^out − slot_{k+1}^in = 0`
/// (`k < K_i − 1`, reachable slots per `anticipated_slot_row_pos`) via
/// [`DeliveryRing::emit_shift_rows`]. A masked slot gets no row — its outgoing column
/// is frozen `[0, 0]` by `fill_anticipated_slot_columns` (the two-sided masking contract).
fn fill_anticipated_slot_definition_entries(
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let row_start = layout.anticipated.row_anticipated_slot_definition_start;
    let ring = anticipated_ring(layout);
    let n_reachable = ring.emit_shift_rows(
        &layout.anticipated.anticipated_slot_row_pos,
        row_start,
        col_entries,
    );
    debug_assert_eq!(
        n_reachable, layout.anticipated.n_anticipated_slot_definition_rows,
        "fill_anticipated_slot_definition_entries: reachable-slot count must match \
         n_anticipated_slot_definition_rows"
    );
}

/// Returns `true` when hydro `h_idx` is in the `PreFilling` phase at this stage.
#[inline]
pub(super) fn is_prefilling(ctx: &TemplateBuildCtx<'_>, stage: &Stage, h_idx: usize) -> bool {
    let hydro = &ctx.hydros[h_idx];
    matches!(
        cobre_core::commissioning::filling_phase(
            hydro.filling.as_ref(),
            hydro.entry_stage_id,
            hydro.exit_stage_id,
            stage.id,
        ),
        cobre_core::commissioning::Phase::PreFilling
    )
}

/// Resolve the cascade target an absent `PreFilling` hydro `h_idx` routes its water
/// onto: the FIRST downstream hydro NOT `PreFilling` at this stage. `None` (SINK) when
/// the chain reaches a terminal, an unresolved id, or stays `PreFilling` all the way
/// down — then `h`'s water exits the system.
///
/// The target MUST be non-`PreFilling`: a `PreFilling` row is the frozen identity
/// `v_d − v_d_in = 0`, and routing any term onto it corrupts that constraint. Routing
/// to the immediate `downstream(h)` unconditionally is the wrong-but-compiling
/// alternative — it corrupts that frozen row when the immediate downstream is itself
/// `PreFilling` (see [`fill_prefilling_shortcircuit`]).
///
/// The `hydros.len()`-bounded loop is defense-in-depth: `check_cascade_acyclic` already
/// proves the walk terminates.
pub(super) fn resolve_shortcircuit_target(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    h_idx: usize,
) -> Option<usize> {
    let mut current_id = ctx.hydros[h_idx].id;
    for _ in 0..ctx.hydros.len() {
        let down_id = ctx.cascade.downstream(current_id)?;
        let d_idx = *ctx.hydro_pos.get(&down_id)?;
        if !is_prefilling(ctx, stage, d_idx) {
            return Some(d_idx);
        }
        current_id = down_id;
    }
    None
}

/// Fill water-balance row entries. Incoming state is pinned via column bounds, so no
/// row-equality state-fixing diagonals are written here.
///
/// A `PreFilling` hydro's row collapses to the frozen-storage identity `v_h − v_h_in = 0`,
/// its water interactions routed to the first non-`PreFilling` downstream by
/// [`fill_prefilling_shortcircuit`] (the contract home). The forbidden alternative —
/// zeroing `h`'s flow columns (turbine/spillage/diversion) while leaving its inflow on
/// this row untouched — traps the water and makes the LP infeasible whenever the site
/// has inflow.
///
/// In `BlockMode::Chronological` the single row becomes `K` chained rows
/// ([`fill_chronological_water_entries`]) that telescope to this parallel row, so
/// `K = 1` is byte-identical to parallel.
pub(super) fn fill_state_and_water_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    // Mode-independent: the bucket ring-shift never depends on block_mode, so it runs
    // once outside the per-mode match.
    fill_transit_bucket_definition_entries(layout, col_entries);

    match stage.block_mode {
        BlockMode::Parallel => {
            fill_parallel_water_entries(ctx, stage, stage_idx, layout, col_entries);
        }
        BlockMode::Chronological => {
            fill_chronological_water_entries(ctx, stage, stage_idx, layout, col_entries);
        }
    }
}

/// Parallel single-row water-balance fill: one equality row per hydro summing all
/// blocks, with per-block flow terms scaled by `τ_k` and the inflow/AR-lag/
/// evaporation/withdrawal families scaled once by the stage total `ζ = Σ_k τ_k`.
fn fill_parallel_water_entries(
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
    let row_water = layout.rows.water_balance.start;
    let col_storage_in_start = layout.col_storage_in_start();
    let col_inflow_lags_start = layout.col_inflow_lags_start();

    for h_idx in 0..n_h {
        let hydro = &ctx.hydros[h_idx];
        let row = row_water + h_idx;

        if is_prefilling(ctx, stage, h_idx) {
            // Frozen-storage identity `v_h − v_h_in = 0`: emit ONLY these two entries.
            // Any inflow/upstream/AR-lag/withdrawal/evaporation coupling left here makes
            // `β_h` stale-nonzero — a wrong cut that still compiles.
            col_entries[h_idx].push((row, 1.0));
            col_entries[col_storage_in_start + h_idx].push((row, -1.0));
            fill_prefilling_shortcircuit(ctx, stage, h_idx, layout, col_entries);
            continue;
        }

        col_entries[h_idx].push((row, 1.0));
        col_entries[col_storage_in_start + h_idx].push((row, -1.0));

        // The maturing-now bucket `b_1^in`: a SINGLE entry — the confluence sum over
        // every upstream arc lives in the state variable itself. Absent with no arc.
        if let Some(range) = plant_transit_bucket_range(layout.state, h_idx) {
            let ring = transit_bucket_ring(layout.state, range);
            col_entries[ring.in_col(0, 0)].push((row, -1.0));
        }

        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col_turbine = layout.turbine_col(HydroSys::new(h_idx), blk);
            col_entries[col_turbine].push((row, tau_h));
            let col_spillage = layout.spillage_col(HydroSys::new(h_idx), blk);
            col_entries[col_spillage].push((row, tau_h));
            let col_diversion = layout.diversion_col(HydroSys::new(h_idx), blk);
            col_entries[col_diversion].push((row, tau_h));
            for &up_id in ctx.cascade.upstream(hydro.id) {
                if let Some(&u_idx) = ctx.hydro_pos.get(&up_id) {
                    fill_arc_release_block_entries(
                        ctx,
                        layout,
                        u_idx,
                        h_idx,
                        stage_idx,
                        blk,
                        tau_h,
                        row,
                        col_entries,
                    );
                }
            }
            if let Some(sources) = ctx.diversion_upstream.get(&hydro.id) {
                for &d_idx in sources {
                    let col_div = layout.diversion_col(HydroSys::new(d_idx), blk);
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

    // The PreFilling `continue`s below keep the frozen identity row free of slack/flow
    // terms (the contract above); `evap_hydro_indices` already excludes PreFilling hydros.
    if ctx.has_penalty {
        for h_idx in 0..n_h {
            if is_prefilling(ctx, stage, h_idx) {
                continue;
            }
            let col = layout.slack.inflow_slack.start + h_idx;
            let row = row_water + h_idx;
            col_entries[col].push((row, -zeta));
        }
    }

    for (local_idx, &h) in layout.evap_hydro_indices.iter().enumerate() {
        let col_evaporation_flow = layout.evap_flow_col(EvapLocal::new(local_idx), 0);
        let row = row_water + h.get();
        col_entries[col_evaporation_flow].push((row, zeta));
    }

    for h_idx in 0..n_h {
        if is_prefilling(ctx, stage, h_idx) {
            continue;
        }
        let col = layout.slack.withdrawal_slack_neg.start + h_idx;
        let row = row_water + h_idx;
        col_entries[col].push((row, -zeta));
    }

    for h_idx in 0..n_h {
        if is_prefilling(ctx, stage, h_idx) {
            continue;
        }
        let col = layout.slack.withdrawal_slack_pos.start + h_idx;
        let row = row_water + h_idx;
        col_entries[col].push((row, zeta));
    }
}

/// Each downstream plant's contiguous bucket sub-range (relative to
/// `transit_buckets_out`/`transit_buckets_in`'s own start), in
/// `transit_bucket_column_order`'s plant-major order.
pub(super) fn transit_bucket_plant_ranges(state: &StateLayout) -> Vec<std::ops::Range<usize>> {
    let order = &state.transit_bucket_column_order;
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < order.len() {
        let plant_idx = order[start].0;
        let mut end = start + 1;
        while end < order.len() && order[end].0 == plant_idx {
            end += 1;
        }
        ranges.push(start..end);
        start = end;
    }
    ranges
}

/// One plant's [`DeliveryRing`] (`n_lanes = 1`) over its LOCAL bucket sub-`range`
/// (relative to `transit_buckets_out`/`transit_buckets_in`'s own start) — the single
/// owner of the ragged-to-dense addressing every bucket call site shares.
pub(super) fn transit_bucket_ring(
    state: &StateLayout,
    range: std::ops::Range<usize>,
) -> DeliveryRing {
    let depth = range.len();
    DeliveryRing::new(
        state.transit_buckets_out.start + range.start..state.transit_buckets_out.start + range.end,
        state.transit_buckets_in.start + range.start..state.transit_buckets_in.start + range.end,
        1,
        depth,
    )
}

/// Fill the travel-time bucket-definition ring-shift rows via
/// [`DeliveryRing::emit_shift_rows`], once per downstream plant. A masked-out bucket
/// gets no row — its outgoing column is frozen `[0, 0]` by `fill_transit_bucket_columns`
/// (the two-sided masking contract). Mode-independent (buckets are stage-level), so it
/// runs once per stage; the deposit terms are emitted separately by
/// [`fill_arc_release_block_entries`]. A no-op when `state.n_buckets == 0`.
fn fill_transit_bucket_definition_entries(
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let state = layout.state;
    let row_start = layout.rows.transit_bucket_definition.start;
    for range in transit_bucket_plant_ranges(state) {
        let ring = transit_bucket_ring(state, range.clone());
        ring.emit_shift_rows(
            &layout.rows.transit_bucket_row_pos[range],
            row_start,
            col_entries,
        );
    }
}

/// One upstream release's per-block contribution to the downstream water balance
/// (arc `u_idx → h_idx`), split by the arc's resolved stage-clock weight `k`: same-stage
/// share `-k_0·τ_blk` on the balance row, and, for a multi-lag arc, deposits `-k_d·τ_blk`
/// (`d = 1..=depth`) into the plant's bucket-definition rows. The SAME release column
/// carries `k_0` on the balance row and `k_1..k_d` into the definition rows — never the
/// once-per-stage `ζ`-family.
///
/// `ctx.arc_stage_weights` has no entry for an undeclared arc, so this emits exactly
/// today's `-τ_blk` and no deposit (the B==0 byte-identity anchor).
fn fill_arc_release_block_entries(
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout,
    u_idx: usize,
    h_idx: usize,
    stage_idx: usize,
    blk: usize,
    tau_h: f64,
    row_balance: usize,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let col_turbine = layout.turbine_col(HydroSys::new(u_idx), blk);
    let col_spillage = layout.spillage_col(HydroSys::new(u_idx), blk);

    let Some(stage_weights) = ctx
        .arc_stage_weights
        .get(&u_idx)
        .map(|k_by_stage| &k_by_stage[stage_idx])
    else {
        col_entries[col_turbine].push((row_balance, -tau_h));
        col_entries[col_spillage].push((row_balance, -tau_h));
        return;
    };

    debug_assert!(
        (stage_weights.iter().sum::<f64>() - 1.0).abs() < 1e-9,
        "arc {u_idx} -> {h_idx} stage {stage_idx}: stage-clock weights must sum to 1.0, got \
         {stage_weights:?}"
    );

    if stage_weights[0] != 0.0 {
        col_entries[col_turbine].push((row_balance, -stage_weights[0] * tau_h));
        col_entries[col_spillage].push((row_balance, -stage_weights[0] * tau_h));
    }

    let depth = stage_weights.len() - 1;
    if depth == 0 {
        return;
    }
    let range = plant_transit_bucket_range(layout.state, h_idx).unwrap_or_else(|| {
        unreachable!(
            "hydro {h_idx} receives a depth-{depth} deposit at stage {stage_idx} but has no \
             bucket range (TransitBucketTopology/arc_stage_weights disagreement)"
        )
    });
    let ring = transit_bucket_ring(layout.state, range.clone());
    let row_transit_bucket_def_start = layout.rows.transit_bucket_definition.start;
    for (d, &stage_weight) in stage_weights.iter().enumerate().skip(1) {
        if stage_weight == 0.0 {
            continue;
        }
        let slot = range.start + ring.slot_target(0, d);
        // A lag beyond this stage's reachable cap has no definition row: the share is
        // dropped, never misdirected onto another lag's row (Terminal credit deferred).
        let Some(pos) = layout.rows.transit_bucket_row_pos[slot] else {
            continue;
        };
        let row_def = row_transit_bucket_def_start + pos;
        col_entries[col_turbine].push((row_def, -stage_weight * tau_h));
        col_entries[col_spillage].push((row_def, -stage_weight * tau_h));
    }
}

/// The bucket sub-range `[start, end)` (relative to `transit_buckets_out`/
/// `transit_buckets_in`'s own start) for downstream plant `plant_idx`, or `None` when
/// it declares no incoming arc.
fn plant_transit_bucket_range(
    state: &StateLayout,
    plant_idx: usize,
) -> Option<std::ops::Range<usize>> {
    let start = state
        .transit_bucket_column_order
        .iter()
        .position(|&(p, _)| p == plant_idx)?;
    let end = state.transit_bucket_column_order[start..]
        .iter()
        .position(|&(p, _)| p != plant_idx)
        .map_or(state.transit_bucket_column_order.len(), |offset| {
            start + offset
        });
    Some(start..end)
}

/// Chronological per-block water-balance fill: each Operating/Filling hydro emits `K`
/// chained rows (block-major `row_water + h·K + (k−1)`), each the parallel row per block
/// with `τ_k` replacing the stage total `ζ` EVERYWHERE. A stray `ζ` double-applies
/// (`Σ_k τ_k = ζ`) and breaks the telescoping identity that recovers the parallel row.
/// The inflow STATE (`z_inflow` and its definition row) stays stage-level.
///
/// A `PreFilling` hydro emits `K` per-block frozen identities `Sᵏ − Sᵏ⁻¹ = 0` (any
/// coupling left on a frozen row makes `β_h` stale-nonzero — a wrong cut that compiles),
/// short-circuiting per block via [`fill_prefilling_shortcircuit`].
fn fill_chronological_water_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_h = layout.n_h;
    let n_blks = layout.n_blks;
    let lag_order = layout.lag_order;
    let row_water = layout.rows.water_balance.start;
    let col_inflow_lags_start = layout.col_inflow_lags_start();
    let has_par = ctx.par_lp.n_stages() > 0 && ctx.par_lp.n_hydros() == n_h;

    for h_idx in 0..n_h {
        let hydro = &ctx.hydros[h_idx];

        if is_prefilling(ctx, stage, h_idx) {
            for k in 1..=n_blks {
                let row = row_water + h_idx * n_blks + (k - 1);
                col_entries[layout.block_storage_col(HydroSys::new(h_idx), k)].push((row, 1.0));
                col_entries[layout.block_storage_col(HydroSys::new(h_idx), k - 1)]
                    .push((row, -1.0));
            }
            fill_prefilling_shortcircuit(ctx, stage, h_idx, layout, col_entries);
            continue;
        }

        // The incoming maturing bucket `b_1^in` delivers over this stage's blocks by the
        // fixed `arrival_density` (fixed-delivery-density contract) — one entry per block.
        if let Some(range) = plant_transit_bucket_range(layout.state, h_idx) {
            let arrival_density =
                resolve_chrono_arrival_density(ctx, stage, stage_idx, hydro.id, n_blks);
            debug_assert!(
                (arrival_density.iter().sum::<f64>() - 1.0).abs() < 1e-9,
                "hydro {h_idx} stage {stage_idx}: arrival_density must sum to 1.0"
            );
            let ring = transit_bucket_ring(layout.state, range);
            let col_first_slot_in = ring.in_col(0, 0);
            for (target_slot, &rho_val) in arrival_density.iter().enumerate() {
                if rho_val == 0.0 {
                    continue;
                }
                let row = row_water + h_idx * n_blks + target_slot;
                col_entries[col_first_slot_in].push((row, -rho_val));
            }
        }

        let psi = has_par.then(|| ctx.par_lp.psi_slice(stage_idx, h_idx));
        for k in 1..=n_blks {
            let blk = k - 1;
            let row = row_water + h_idx * n_blks + blk;
            let tau_k = stage.blocks[blk].duration_hours * M3S_TO_HM3;

            col_entries[layout.block_storage_col(HydroSys::new(h_idx), k)].push((row, 1.0));
            col_entries[layout.block_storage_col(HydroSys::new(h_idx), k - 1)].push((row, -1.0));

            col_entries[layout.turbine_col(HydroSys::new(h_idx), blk)].push((row, tau_k));
            col_entries[layout.spillage_col(HydroSys::new(h_idx), blk)].push((row, tau_k));
            col_entries[layout.diversion_col(HydroSys::new(h_idx), blk)].push((row, tau_k));
            for &up_id in ctx.cascade.upstream(hydro.id) {
                if let Some(&u_idx) = ctx.hydro_pos.get(&up_id) {
                    fill_arc_release_chrono_block_entries(
                        ctx,
                        layout,
                        stage,
                        u_idx,
                        h_idx,
                        stage_idx,
                        blk,
                        row_water,
                        col_entries,
                    );
                }
            }
            if let Some(sources) = ctx.diversion_upstream.get(&hydro.id) {
                for &d_idx in sources {
                    col_entries[layout.diversion_col(HydroSys::new(d_idx), blk)]
                        .push((row, -tau_k));
                }
            }

            if let Some(psi) = psi {
                for (lag, &psi_val) in psi.iter().enumerate() {
                    if psi_val != 0.0 && lag < lag_order {
                        let col = col_inflow_lags_start + lag * n_h + h_idx;
                        col_entries[col].push((row, -tau_k * psi_val));
                    }
                }
            }

            if ctx.has_penalty {
                col_entries[layout.slack.inflow_slack.start + h_idx].push((row, -tau_k));
            }
            col_entries[layout.slack.withdrawal_slack_neg.start + h_idx].push((row, -tau_k));
            col_entries[layout.slack.withdrawal_slack_pos.start + h_idx].push((row, tau_k));
        }
    }

    for (local_idx, &h) in layout.evap_hydro_indices.iter().enumerate() {
        let local_idx = EvapLocal::new(local_idx);
        for k in 1..=n_blks {
            let blk = k - 1;
            let tau_k = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let row = row_water + h.get() * n_blks + blk;
            col_entries[layout.evap_flow_col(local_idx, blk)].push((row, tau_k));
        }
    }
}

/// One upstream release's per-block contribution to the downstream chained rows
/// (arc `u_idx → h_idx`, block `blk`): same-stage routing `-κ·τ_blk` onto downstream
/// blocks and crossing deposits `-χ·τ_blk` into the plant's bucket-definition rows — the
/// SAME release column carrying both. Verifies the shared-density aggregation identity
/// `Σ_b w_b·χ_{b,d} == k_d` once per (arc, stage) (`blk == 0`).
///
/// `arc_spread_chrono` has no entry for an undeclared arc (or a `Parallel`-mode stage),
/// so this emits today's `-τ_blk` and no routing/deposit — the B==0/K==1 byte-identity
/// anchor.
fn fill_arc_release_chrono_block_entries(
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout,
    stage: &Stage,
    u_idx: usize,
    h_idx: usize,
    stage_idx: usize,
    blk: usize,
    row_water: usize,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let row_base = row_water + h_idx * n_blks;
    let tau_k = stage.blocks[blk].duration_hours * M3S_TO_HM3;
    let col_turbine = layout.turbine_col(HydroSys::new(u_idx), blk);
    let col_spillage = layout.spillage_col(HydroSys::new(u_idx), blk);

    let Some(resolution) = ctx
        .arc_spread_chrono
        .get(&u_idx)
        .and_then(|by_stage| by_stage[stage_idx].as_ref())
    else {
        col_entries[col_turbine].push((row_base + blk, -tau_k));
        col_entries[col_spillage].push((row_base + blk, -tau_k));
        return;
    };

    let block_routing = &resolution.within_stage_routing[blk];
    let block_deposit = &resolution.block_deposits[blk];
    debug_assert!(
        (block_routing.iter().sum::<f64>() + block_deposit[1..].iter().sum::<f64>() - 1.0).abs()
            < 1e-9,
        "arc {u_idx} block {blk} stage {stage_idx}: per-column conservation \
         sum(within_stage_routing) + sum(block_deposits[1..]) must equal 1.0"
    );
    if blk == 0 {
        debug_assert!(
            (resolution.stage_weights.iter().sum::<f64>() - 1.0).abs() < 1e-9,
            "arc {u_idx} stage {stage_idx}: stage-clock weights must sum to 1.0, got {:?}",
            resolution.stage_weights
        );
        for (d, &stage_weight) in resolution.stage_weights.iter().enumerate() {
            let aggregated: f64 = resolution
                .block_deposits
                .iter()
                .zip(&stage.blocks)
                .map(|(deposit_row, b)| {
                    (b.duration_hours * M3S_TO_HM3 / layout.zeta) * deposit_row[d]
                })
                .sum();
            debug_assert!(
                (aggregated - stage_weight).abs() < 1e-9,
                "arc {u_idx} stage {stage_idx}: block deposits must aggregate to k_d (d={d})"
            );
        }
    }

    for (j, &routing_val) in block_routing.iter().enumerate() {
        if routing_val == 0.0 {
            continue;
        }
        let row = row_base + blk + j;
        col_entries[col_turbine].push((row, -routing_val * tau_k));
        col_entries[col_spillage].push((row, -routing_val * tau_k));
    }

    let depth = block_deposit.len() - 1;
    if depth == 0 {
        return;
    }
    let range = plant_transit_bucket_range(layout.state, h_idx).unwrap_or_else(|| {
        unreachable!(
            "hydro {h_idx} receives a depth-{depth} deposit at stage {stage_idx} but has no \
             bucket range (TransitBucketTopology/arc_spread_chrono disagreement)"
        )
    });
    let ring = transit_bucket_ring(layout.state, range.clone());
    let row_transit_bucket_def_start = layout.rows.transit_bucket_definition.start;
    for (d, &deposit_d) in block_deposit.iter().enumerate().skip(1) {
        if deposit_d == 0.0 {
            continue;
        }
        let slot = range.start + ring.slot_target(0, d);
        // A lag beyond this stage's reachable cap has no row to deposit into (dropped,
        // never misdirected — Terminal credit deferred).
        let Some(pos) = layout.rows.transit_bucket_row_pos[slot] else {
            continue;
        };
        let row_def = row_transit_bucket_def_start + pos;
        col_entries[col_turbine].push((row_def, -deposit_d * tau_k));
        col_entries[col_spillage].push((row_def, -deposit_d * tau_k));
    }
}

/// Resolve this stage's incoming maturing bucket `arrival_density` (fixed-delivery-density
/// contract): a lookup of the setup-precomputed per-`(arc, arrival stage)` blend
/// ([`build_arc_arrival_density`](crate::setup::bucket_topology::build_arc_arrival_density)),
/// already resolved in this arrival stage's own frame. Falls back to duration-weighted
/// uniform only where the table holds no blend (the study's first stage) or the plant has
/// no travel-time upstream.
///
/// A non-travel-time upstream is EXCLUDED, never folded in via `uniform`: it would
/// disagree with the sole travel-time arc's non-uniform density — a false
/// heterogeneous-confluence panic in debug, a silent uniform split in release.
///
/// A heterogeneous-density confluence has no resolved policy;
/// `check_chronological_confluence_heterogeneous_travel_time` (`cobre-io`) rejects it at
/// config time, so the `debug_assert!` below is a defensive backstop, not the enforcement
/// point.
fn resolve_chrono_arrival_density(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    downstream_id: EntityId,
    n_blks: usize,
) -> Vec<f64> {
    let uniform = || {
        let total: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
        stage
            .blocks
            .iter()
            .map(|b| b.duration_hours / total)
            .collect::<Vec<f64>>()
    };

    let mut chosen: Option<Vec<f64>> = None;
    for &up_id in ctx.cascade.upstream(downstream_id) {
        let Some(&u_idx) = ctx.hydro_pos.get(&up_id) else {
            continue;
        };
        let Some(by_stage) = ctx.arc_arrival_density.get(&u_idx) else {
            continue;
        };
        let candidate = by_stage[stage_idx].clone().map_or_else(uniform, |density| {
            debug_assert_eq!(
                density.len(),
                n_blks,
                "arc {u_idx} stage {stage_idx}: arrival_density length must equal n_blks"
            );
            density
        });
        match &chosen {
            None => chosen = Some(candidate),
            Some(existing) => {
                debug_assert!(
                    existing.len() == candidate.len()
                        && existing
                            .iter()
                            .zip(&candidate)
                            .all(|(&a, &b)| (a - b).abs() < 1e-9),
                    "confluence with heterogeneous chronological delivery densities into \
                     one downstream plant is not yet supported (arc {u_idx} disagrees at \
                     stage {stage_idx})"
                );
            }
        }
    }
    chosen.unwrap_or_else(uniform)
}

/// Re-route an absent `PreFilling` hydro `h`'s water interactions onto the FIRST
/// non-`PreFilling` downstream hydro `d` ([`resolve_shortcircuit_target`]): in `Parallel`
/// onto `d`'s single row, in `Chronological` onto `d`'s block rows with the stage-total
/// `−ζ` on `z_h` split into `−τ_k` per block (`Σ_k τ_k = ζ`).
///
/// Route the REAL `ζ`/`τ_k` columns, never a synthesized coefficient: with the real
/// columns on `d`'s row, `rc / col_scale` of `d`'s pinned incoming-storage column is a
/// valid subgradient; a synthesized coefficient breaks that duality and produces an
/// invalid cut.
///
/// `z_h`'s own definition row and noise patch are left UNTOUCHED for `h`, so the routed
/// column is scenario-exact; the deterministic base rides on `z_h` and must NOT also be
/// added to `d`'s RHS (double-count). `h`'s withdrawal demand transfers via `d`'s RHS
/// (owned by `super::rows::fill_water_balance_rows`), which MUST resolve the SAME `d` so
/// matrix and RHS agree. A skipped intermediate `PreFilling` hydro contributes zero
/// releases, so a chain re-routes each link's inflow to `d` exactly once. Sink case (no
/// non-`PreFilling` downstream): nothing routed, no panic.
fn fill_prefilling_shortcircuit(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    h_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let hydro = &ctx.hydros[h_idx];
    let Some(d_idx) = resolve_shortcircuit_target(ctx, stage, h_idx) else {
        return;
    };
    let n_blks = layout.n_blks;
    let row_water = layout.rows.water_balance.start;
    let z_h = layout.col_z_inflow_start() + h_idx;

    let row_d_for = |blk: usize| match stage.block_mode {
        BlockMode::Parallel => row_water + d_idx,
        BlockMode::Chronological => row_water + d_idx * n_blks + blk,
    };

    // Parallel pushes the single stage-total `−ζ` (a `Σ_k −τ_k` loop would inflate
    // `z_h`'s routed-entry count, pinned by
    // `prefilling_upstream_inflow_lands_on_balance_row_only`); Chronological splits into
    // per-block `−τ_k` (`Σ_k τ_k = ζ`).
    match stage.block_mode {
        BlockMode::Parallel => col_entries[z_h].push((row_water + d_idx, -layout.zeta)),
        BlockMode::Chronological => {
            for blk in 0..n_blks {
                let tau_k = stage.blocks[blk].duration_hours * M3S_TO_HM3;
                col_entries[z_h].push((row_water + d_idx * n_blks + blk, -tau_k));
            }
        }
    }

    for blk in 0..n_blks {
        let tau_k = stage.blocks[blk].duration_hours * M3S_TO_HM3;
        let row_d = row_d_for(blk);
        for &up_id in ctx.cascade.upstream(hydro.id) {
            if let Some(&u_idx) = ctx.hydro_pos.get(&up_id) {
                col_entries[layout.turbine_col(HydroSys::new(u_idx), blk)].push((row_d, -tau_k));
                col_entries[layout.spillage_col(HydroSys::new(u_idx), blk)].push((row_d, -tau_k));
            }
        }
        if let Some(sources) = ctx.diversion_upstream.get(&hydro.id) {
            for &src_idx in sources {
                col_entries[layout.diversion_col(HydroSys::new(src_idx), blk)]
                    .push((row_d, -tau_k));
            }
        }
    }
}

/// Fill the LHS of the per-stage soft filling-target row `v_h + σ_fill ≥ V_target[t]`
/// for each Filling-phase hydro: `+1.0` on the outgoing storage column `v_h` and `+1.0`
/// on the `σ_fill` slack. The `≥` sense and RHS are set by
/// [`super::rows::fill_filling_target_rows`].
///
/// Cut validity: LP duality folds the `σ_fill` soft-row dual into the incoming-storage
/// column's `rc / col_scale`. NEVER separately extract this row's dual and add it by hand
/// — that double-counts the soft floor (a guard test asserts no `lp/builder` file
/// references the dual-extraction entry point).
fn fill_filling_target_entries(layout: &StageLayout, col_entries: &mut [Vec<(usize, f64)>]) {
    let row_start = layout.filling.row_filling_target_start;
    let col_start = layout.filling.col_filling_target_start;
    for (local_idx, &h) in layout
        .filling
        .filling_target_hydro_indices
        .iter()
        .enumerate()
    {
        let row = row_start + local_idx;
        col_entries[h.get()].push((row, 1.0));
        col_entries[col_start + local_idx].push((row, 1.0));
    }
}

/// Fill the LHS of the soft operating-floor row `v_h + σ^{v-} ≥ min_storage_hm3` for
/// each Operating-phase filling hydro: `+1.0` on `v_h` and `+1.0` on the `σ^{v-}` slack.
/// The `≥` sense and RHS are set by [`super::rows::fill_filled_min_storage_floor_rows`].
///
/// Same cut-validity contract as [`fill_filling_target_entries`]: never hand-extract this
/// row's dual. DISTINCT from that sibling — different slack, non-overlapping stage scope
/// (Operating vs Filling), different RHS and cost; never conflate them.
fn fill_filled_min_storage_floor_entries(
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let row_start = layout.filling.row_filled_min_storage_floor_start;
    let col_start = layout.filling.col_filled_min_storage_floor_start;
    for (local_idx, &h) in layout
        .filling
        .filled_min_storage_floor_hydro_indices
        .iter()
        .enumerate()
    {
        let row = row_start + local_idx;
        col_entries[h.get()].push((row, 1.0));
        col_entries[col_start + local_idx].push((row, 1.0));
    }
}

/// Fill pumping-flow water-balance entries: per block, the pumped-flow column enters the
/// SOURCE hydro's water row with `+tau_h` (outflow sign) and the DESTINATION's with
/// `−tau_h` (inflow sign). `tau_h` is the identical `duration_hours * M3S_TO_HM3`
/// expression turbine/spillage use, so the coefficient stays bit-identical across sites.
/// Structural entries are written for every station: a dormant station's column is `[0, 0]`.
pub(super) fn fill_pumping_water_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let grid = layout.block_grid();
    let row_water = layout.rows.water_balance.start;
    for (p_sys, station) in ctx.pumping_stations.iter().enumerate() {
        // Per-side guards are defense-in-depth (`validate_pumping_station_refs` guarantees
        // resolution on a production `System`). Do NOT promote to an unconditional
        // index/expect — a one-sided resolve writes a feasible-but-wrong half coupling.
        let source = ctx.hydro_pos.get(&station.source_hydro_id).copied();
        let destination = ctx.hydro_pos.get(&station.destination_hydro_id).copied();
        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col = grid.flat(layout.equipment.col_pumping_start, p_sys, blk);
            if let Some(s_idx) = source {
                col_entries[col].push((row_water + s_idx, tau_h));
            }
            if let Some(d_idx) = destination {
                col_entries[col].push((row_water + d_idx, -tau_h));
            }
        }
    }
}

/// Fill load-balance entries for hydro/thermal generation, line flows, pumping power,
/// and deficit/excess slacks. FPHA hydros enter with `g_{h,k}` at `+1.0`;
/// constant-productivity hydros with `rho * turbine_col`.
///
/// Pumping power is a negative injection: the `pumping_flow` column enters with
/// `−consumption_mw_per_m3s` (no separate power column). A positive coefficient would
/// credit the bus for power the station consumes.
pub(super) fn fill_load_balance_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let grid = layout.block_grid();
    let row_load = layout.rows.load_balance.start;

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
                    let row = grid.flat(row_load, b_idx, blk);
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
                    let row = grid.flat(row_load, b_idx, blk);
                    let col = layout.turbine_col(HydroSys::new(h_idx), blk);
                    col_entries[col].push((row, rho));
                }
            }
        }
    }

    for (t_idx, thermal) in ctx.thermals.iter().enumerate() {
        if let Some(&b_idx) = ctx.bus_pos.get(&thermal.bus_id) {
            for blk in 0..n_blks {
                let row = grid.flat(row_load, b_idx, blk);
                let col = grid.flat(layout.equipment.thermal.start, t_idx, blk);
                col_entries[col].push((row, 1.0));
            }
        }
    }

    for (l_idx, line) in ctx.lines.iter().enumerate() {
        let src_idx = ctx.bus_pos.get(&line.source_bus_id).copied();
        let tgt_idx = ctx.bus_pos.get(&line.target_bus_id).copied();
        for blk in 0..n_blks {
            let col_fwd = layout.line_fwd_col(LineSys::new(l_idx), blk);
            let col_rev = layout.line_rev_col(LineSys::new(l_idx), blk);
            if let Some(tgt) = tgt_idx {
                let row = grid.flat(row_load, tgt, blk);
                col_entries[col_fwd].push((row, 1.0));
                col_entries[col_rev].push((row, -1.0));
            }
            if let Some(src) = src_idx {
                let row = grid.flat(row_load, src, blk);
                col_entries[col_fwd].push((row, -1.0));
                col_entries[col_rev].push((row, 1.0));
            }
        }
    }

    // Written for every station: a dormant station's pumping column is `[0, 0]`.
    for (p_sys, station) in ctx.pumping_stations.iter().enumerate() {
        if let Some(&b_idx) = ctx.bus_pos.get(&station.bus_id) {
            for blk in 0..n_blks {
                let row = grid.flat(row_load, b_idx, blk);
                let col = grid.flat(layout.equipment.col_pumping_start, p_sys, blk);
                col_entries[col].push((row, -station.consumption_mw_per_m3s));
            }
        }
    }

    // Import injects into its bus (`+1`), export withdraws (`−1`) — INDEPENDENT of the
    // stored price sign; flipping it would make an export feed the bus. Written for every
    // contract: a dormant contract's column is `[0, 0]`.
    for (c_sys, contract) in ctx.contracts.iter().enumerate() {
        let (contract_type, family_slot) =
            crate::generic_constraints::contract_family_slot(ctx.contracts, c_sys);
        let (base, sign) = match contract_type {
            ContractType::Import => (layout.equipment.col_contract_import_start, 1.0),
            ContractType::Export => (layout.equipment.col_contract_export_start, -1.0),
        };
        if let Some(&b_idx) = ctx.bus_pos.get(&contract.bus_id) {
            for blk in 0..n_blks {
                let row = grid.flat(row_load, b_idx, blk);
                let col = grid.flat(base, family_slot, blk);
                col_entries[col].push((row, sign));
            }
        }
    }

    for (b_idx, bus) in ctx.buses.iter().enumerate() {
        for blk in 0..n_blks {
            let row = grid.flat(row_load, b_idx, blk);
            for seg_idx in 0..bus.deficit_segments.len() {
                let col_def = layout.deficit_col(b_idx, seg_idx, blk);
                col_entries[col_def].push((row, 1.0));
            }
            let col_exc = grid.flat(layout.equipment.excess.start, b_idx, blk);
            col_entries[col_exc].push((row, -1.0));
        }
    }
}

/// Fill FPHA hyperplane constraint entries, one row per `(FPHA hydro, block, plane)`,
/// implementing `g − γᵥ/2·v − γᵥ/2·v_in − γ_q·q − γ_s·s ≤ γ₀` (`γ₀` in the row upper
/// bound set by [`super::rows::fill_fpha_rows`]).
///
/// FPHA uses AVERAGE storage `(V_in + V_out)/2`, so `−γᵥ/2` lands on BOTH the outgoing-
/// and incoming-storage columns. Putting it on `V_out` alone compiles and passes
/// single-plane tests but understates generation by the `V_in` head term — the
/// wrong-but-compiling alternative deterministic case D06 pins against.
///
/// In `BlockMode::Chronological` block `k` averages the block-local boundaries
/// `(Sᵏ⁻¹, Sᵏ)`; `K = 1` resolves both back to `(S⁰, Sᴷ)`, byte-identical to parallel.
///
/// Driven by [`for_each_fpha_plane`] so entries and the row bounds set by
/// [`super::rows::fill_fpha_rows`] share one row cursor.
pub(super) fn fill_fpha_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    for_each_fpha_plane(
        ctx,
        stage_idx,
        layout,
        |local_idx, h_idx, blk, _p_idx, plane, row| {
            let (col_v_in, col_v) = match stage.block_mode {
                BlockMode::Parallel => (
                    layout.block_storage_col(h_idx, 0),
                    layout.block_storage_col(h_idx, layout.n_blks),
                ),
                BlockMode::Chronological => (
                    layout.block_storage_col(h_idx, blk),
                    layout.block_storage_col(h_idx, blk + 1),
                ),
            };
            let col_q = layout.turbine_col(h_idx, blk);
            let col_s = layout.spillage_col(h_idx, blk);
            let col_g = layout.generation_col(local_idx, blk);

            col_entries[col_g].push((row, 1.0));
            // Average storage: −γᵥ/2 on BOTH storage columns, not one alone (D06).
            col_entries[col_v_in].push((row, -plane.gamma_v / 2.0));
            col_entries[col_v].push((row, -plane.gamma_v / 2.0));
            col_entries[col_q].push((row, -plane.gamma_q));
            col_entries[col_s].push((row, -plane.gamma_s));
        },
    );
}

/// Fill the evaporation equality rows, one per `(evaporation hydro, block)`, encoding
/// `evaporation_flow − slope/2·Sᵏ⁻¹ − slope/2·Sᵏ + f_plus − f_minus = intercept_m3s`
/// (`slope` = `volume_slope_m3s_per_hm3`; `intercept_m3s` set by `super::rows::fill_stage_rows`).
///
/// Like FPHA, `slope/2` lands on BOTH storage columns to average the block-local storage
/// `(Sᵏ⁻¹ + Sᵏ)/2`; chronological `K = 1` resolves both boundaries back to `(S⁰, Sᴷ)`,
/// byte-identical to parallel.
///
/// The evaporation flow's entry INTO the water-balance row lives with the water-balance
/// fill, not here.
pub(super) fn fill_evaporation_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let row_evap_start = layout.row_evap_start();

    for (local_idx, &h) in layout.evap_hydro_indices.iter().enumerate() {
        let coeff = match ctx.evaporation_models.model(h.get()) {
            EvaporationModel::Linearized { coefficients, .. } => {
                debug_assert!(
                    stage_idx < coefficients.len(),
                    "evap_hydro_indices contains hydro {} but coefficients length {} \
                     is less than stage_idx {}",
                    h.get(),
                    coefficients.len(),
                    stage_idx
                );
                match coefficients.get(stage_idx) {
                    Some(c) => *c,
                    None => continue,
                }
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

        let half_slope = coeff.volume_slope_m3s_per_hm3 / 2.0;
        for k in 1..=n_blks {
            let blk = k - 1;
            let (col_v_in, col_v) = match stage.block_mode {
                BlockMode::Parallel => (
                    layout.block_storage_col(h, 0),
                    layout.block_storage_col(h, layout.n_blks),
                ),
                BlockMode::Chronological => (
                    layout.block_storage_col(h, k - 1),
                    layout.block_storage_col(h, k),
                ),
            };
            let local = EvapLocal::new(local_idx);
            let col_evaporation_flow = layout.evap_flow_col(local, blk);
            let col_f_plus = layout.evap_f_plus_col(local, blk);
            let col_f_minus = layout.evap_f_minus_col(local, blk);
            let row = row_evap_start + local_idx * n_blks + blk;

            col_entries[col_evaporation_flow].push((row, 1.0));
            col_entries[col_v_in].push((row, -half_slope));
            col_entries[col_v].push((row, -half_slope));
            col_entries[col_f_plus].push((row, 1.0));
            col_entries[col_f_minus].push((row, -1.0));
        }
    }
}

/// Mutable LP matrix buffers for stage template construction.
pub(super) struct LpMatrixBuffers<'a> {
    /// CSC column entries (column index -> list of (row, coefficient)).
    pub(super) col_entries: &'a mut [Vec<(usize, f64)>],
    pub(super) col_upper: &'a mut [f64],
    pub(super) objective: &'a mut [f64],
    pub(super) row_lower: &'a mut [f64],
    pub(super) row_upper: &'a mut [f64],
}

/// Fill matrix entries, row bounds, and slack columns for every active generic constraint
/// row at this stage, resolving each term via `resolve_variable_ref` and pricing enabled
/// slack at `penalty * block_hours`. Unknown entity IDs resolve to zero contributions
/// (the defense-in-depth fallback for referential-validation gaps).
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
    if layout.rows.n_generic_rows == 0 {
        return;
    }

    let geom = layout.resolver_geom();
    let positions = crate::generic_constraints::EntityPositionMaps {
        hydro: &ctx.hydro_pos,
        thermal: &ctx.thermal_pos,
        bus: &ctx.bus_pos,
        line: &ctx.line_pos,
    };
    let cascade_refs = crate::generic_constraints::CascadeRefs {
        cascade: ctx.cascade,
        diversion_upstream: &ctx.diversion_upstream,
    };
    let pumping_refs = crate::generic_constraints::PumpingRefs {
        col_pumping_start: layout.equipment.col_pumping_start,
        pumping_stations: ctx.pumping_stations,
        pumping_pos: &ctx.pumping_pos,
    };
    let contract_refs = crate::generic_constraints::ContractRefs {
        contracts: ctx.contracts,
        contract_pos: &ctx.contract_pos,
    };

    for (entry_idx, entry) in layout.generic_constraint_rows.iter().enumerate() {
        let row = layout.rows.row_generic_start + entry_idx;
        let constraint = &ctx.generic_constraints[entry.constraint_idx];
        // A collapsed stage-level row is priced by the stage's total hours (it stands in
        // for one row per block); the total is penalty-conserving either way.
        let block_hours = if entry.is_stage_level {
            stage.blocks.iter().map(|b| b.duration_hours).sum()
        } else {
            stage.blocks[entry.block_idx].duration_hours
        };

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

        for term in &constraint.expression.terms {
            let pairs = resolve_variable_ref(
                &term.variable,
                entry.block_idx,
                stage_idx,
                &geom,
                ctx.production_models,
                &positions,
                &cascade_refs,
                &pumping_refs,
                &contract_refs,
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

        if let Some(plus_col) = entry.slack_plus_col {
            let penalty = constraint.slack.penalty.unwrap_or(0.0);
            let obj_coeff = penalty * block_hours;

            col_upper[plus_col] = f64::INFINITY;
            objective[plus_col] = obj_coeff;

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

            if let Some(minus_col) = entry.slack_minus_col {
                col_upper[minus_col] = f64::INFINITY;
                objective[minus_col] = obj_coeff;
                // LHS + s_g_plus - s_g_minus == bound  →  minus slack with -1.0
                col_entries[minus_col].push((row, -1.0));
            }
        }
    }
}

/// Inject `+1.0` for each NCS at its connected bus's load-balance row, per block. Written
/// for every NCS: a dormant NCS's generation column is `[0, 0]`.
pub(super) fn fill_ncs_load_balance_entries(
    ctx: &TemplateBuildCtx<'_>,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let grid = layout.block_grid();
    for (ncs_sys_idx, ncs) in ctx.non_controllable_sources.iter().enumerate() {
        let Some(&bus_idx) = ctx.bus_pos.get(&ncs.bus_id) else {
            continue;
        };
        for blk in 0..layout.n_blks {
            let col = grid.flat(layout.equipment.col_ncs_start, ncs_sys_idx, blk);
            let row = grid.flat(layout.rows.load_balance.start, bus_idx, blk);
            col_entries[col].push((row, 1.0));
        }
    }
}

/// Fill the z-inflow definition row per hydro `z_h − Σ_l ψ_l·lag_in[h,l] = base_h + σ_h·η_h`:
/// `+1.0` on `z_h`, `−ψ_l` on each nonzero lag column. The lag layout is lag-major
/// (`inflow_lags.start + lag * n_h + h`), matching the water-balance AR-dynamics entries.
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
        let row = layout.rows.z_inflow_row_start + h_idx;

        let col_z = layout.col_z_inflow_start() + h_idx;
        col_entries[col_z].push((row, 1.0));

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

/// Fill entries for the 4 operational-violation families, linking decision
/// variables to their slack columns:
///
/// - **Min outflow** (`>=`): `q + s + d + sigma_below`
/// - **Max outflow** (`<=`): `q + s + d - sigma_above`
/// - **Min turbine** (`>=`): `q + sigma_below`
/// - **Min generation** (`>=`): `var + sigma_below`, where `var` is `rho * q` for
///   constant-productivity hydros or the generation column `g` for FPHA.
pub(super) fn fill_operational_violation_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let grid = layout.block_grid();

    for (h_idx, fpha_local_entry) in layout.fpha_local_index.iter().enumerate() {
        for blk in 0..n_blks {
            let row = grid.flat(
                layout.slack.oper_violation.min_outflow_rows.start,
                h_idx,
                blk,
            );
            let col_q = layout.turbine_col(HydroSys::new(h_idx), blk);
            col_entries[col_q].push((row, 1.0));
            let col_s = layout.spillage_col(HydroSys::new(h_idx), blk);
            col_entries[col_s].push((row, 1.0));
            let col_d = layout.diversion_col(HydroSys::new(h_idx), blk);
            col_entries[col_d].push((row, 1.0));
            let col_slack = layout.outflow_below_col(HydroSys::new(h_idx), blk);
            col_entries[col_slack].push((row, 1.0));
        }

        for blk in 0..n_blks {
            let row = grid.flat(
                layout.slack.oper_violation.max_outflow_rows.start,
                h_idx,
                blk,
            );
            let col_q = layout.turbine_col(HydroSys::new(h_idx), blk);
            col_entries[col_q].push((row, 1.0));
            let col_s = layout.spillage_col(HydroSys::new(h_idx), blk);
            col_entries[col_s].push((row, 1.0));
            let col_d = layout.diversion_col(HydroSys::new(h_idx), blk);
            col_entries[col_d].push((row, 1.0));
            let col_slack = layout.outflow_above_col(HydroSys::new(h_idx), blk);
            col_entries[col_slack].push((row, -1.0));
        }

        for blk in 0..n_blks {
            let row = grid.flat(
                layout.slack.oper_violation.min_turbine_rows.start,
                h_idx,
                blk,
            );
            let col_q = layout.turbine_col(HydroSys::new(h_idx), blk);
            col_entries[col_q].push((row, 1.0));
            let col_slack = layout.turbine_below_col(HydroSys::new(h_idx), blk);
            col_entries[col_slack].push((row, 1.0));
        }

        if let Some(&local_fpha_idx) = fpha_local_entry.as_ref() {
            for blk in 0..n_blks {
                let row = grid.flat(
                    layout.slack.oper_violation.min_generation_rows.start,
                    h_idx,
                    blk,
                );
                let col_g = layout.generation_col(local_fpha_idx, blk);
                col_entries[col_g].push((row, 1.0));
                let col_slack = layout.generation_below_col(HydroSys::new(h_idx), blk);
                col_entries[col_slack].push((row, 1.0));
            }
        } else {
            // Read rho from the resolved per-stage production model, not a static field.
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
                let row = grid.flat(
                    layout.slack.oper_violation.min_generation_rows.start,
                    h_idx,
                    blk,
                );
                let col_q = layout.turbine_col(HydroSys::new(h_idx), blk);
                col_entries[col_q].push((row, rho));
                let col_slack = layout.generation_below_col(HydroSys::new(h_idx), blk);
                col_entries[col_slack].push((row, 1.0));
            }
        }
    }
}

/// Build the unsorted CSC matrix entries for one stage: one `Vec<(row, value)>` per column
/// in insertion order. The caller must sort by row index before assembling (see
/// `build_single_stage_template`; `assemble_csc` asserts the sort).
pub(super) fn build_stage_matrix_entries(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
    layout: &StageLayout,
) -> Vec<Vec<(usize, f64)>> {
    let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];

    fill_state_and_water_entries(ctx, stage, stage_idx, layout, &mut col_entries);
    fill_filling_target_entries(layout, &mut col_entries);
    fill_filled_min_storage_floor_entries(layout, &mut col_entries);
    fill_pumping_water_entries(ctx, stage, layout, &mut col_entries);
    fill_anticipated_state_out_def_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_anticipated_slot_definition_entries(layout, &mut col_entries);
    fill_load_balance_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_ncs_load_balance_entries(ctx, layout, &mut col_entries);
    fill_fpha_entries(ctx, stage, stage_idx, layout, &mut col_entries);
    fill_evaporation_entries(ctx, stage, stage_idx, layout, &mut col_entries);
    fill_z_inflow_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_operational_violation_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_anticipated_fishing_entries(ctx, stage, layout, &mut col_entries);

    col_entries
}

/// Assemble CSC arrays from per-column entry lists.
///
/// Returns `(col_starts, row_indices, values)` in the format required by
/// `SolverInterface::load_model`.
pub(super) fn assemble_csc(col_entries: &[Vec<(usize, f64)>]) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    // Caller owns the sort; `assemble_csc` does NOT sort. Unsorted `row_indices` within a
    // column can make HiGHS/CLP silently misfactorize, so this debug_assert surfaces a
    // missing caller-side sort rather than masking it with a re-sort.
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
            // Rationale: the stage LP row count is far below i32::MAX, so the
            // i32 cast the HiGHS/CLP C API demands cannot truncate or wrap.
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            row_indices.push(row as i32);
            values.push(val);
        }
        // Rationale: the running nonzero offset is far below i32::MAX, so the i32
        // offset the solver C API demands cannot overflow.
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
        ScalarParameter, SlackConfig, StageId, SystemBuilder, ThermalStageBounds,
    };
    use cobre_core::{LinearTerm, VariableRef};
    use cobre_stochastic::normal::precompute::PrecomputedNormal;
    use cobre_stochastic::par::precompute::PrecomputedPar;
    use std::collections::HashMap;

    use crate::energy_conversion::{EnergyConversionSet, build_hydro_energy_productivity_override};
    use crate::hydro_models::PrepareHydroModelsResult;
    use crate::inflow_method::InflowNonNegativityMethod;
    use crate::resolved_parameters::build_resolved_parameters;

    /// `StageId(0)..StageId(n_stages - 1)`: the 0-based domain ids these
    /// fixtures use (no `Computed` parameter reads the override table here).
    fn stage_ids_0_based(n_stages: usize) -> Vec<StageId> {
        (0..n_stages)
            .map(|s| StageId(i32::try_from(s).expect("test stage count fits in i32")))
            .collect()
    }

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
            filling_min_rate_m3s: 0.0,
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
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let thermal = Thermal {
            id: thermal_entity_id,
            name: "T1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
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
        crate::lp_builder::build_stage_templates_resolving_layout(
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
        let stage_ids = stage_ids_0_based(n_stages);
        let ec = EnergyConversionSet::new(vec![], vec![], 0, n_stages);
        let override_table =
            build_hydro_energy_productivity_override(&[]).expect("empty override table");
        build_resolved_parameters(
            &[],
            &ec,
            &override_table,
            &[],
            &stage_to_season,
            &stage_ids,
            n_stages,
        )
        .expect("empty_resolved_params: valid")
    }

    /// Build a `ResolvedParameters` table containing a single `Constant` parameter.
    fn constant_param_resolved(
        param_id: EntityId,
        value: f64,
        n_stages: usize,
    ) -> crate::resolved_parameters::ResolvedParameters {
        let stage_to_season: Vec<i32> = vec![0; n_stages];
        let stage_ids = stage_ids_0_based(n_stages);
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
            &stage_ids,
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
        let stage_ids = stage_ids_0_based(n_stages);
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
            &stage_ids,
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
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names
)]
mod zero_cost_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, CascadeTopology, ContractStageBounds, HydroStageBounds,
        LineStageBounds, PumpingStageBounds, ResolvedBounds, ResolvedExchangeFactors,
        ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds,
        ResolvedNcsFactors, ResolvedPenalties, Stage, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{EvaporationModelSet, ProductionModelSet};
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::columns::{ColumnBufs, fill_stage_columns, fill_thermal_columns};
    use super::super::layout::{ResolvedTables, StageLayout, TemplateBuildCtx};
    use super::super::rows::{
        fill_anticipated_fishing_rows, fill_anticipated_state_out_def_rows, fill_stage_rows,
    };
    use super::super::test_support::{state_layout_for, two_block_stage};
    use super::{
        build_stage_matrix_entries, fill_anticipated_fishing_entries,
        fill_anticipated_slot_definition_entries, fill_anticipated_state_out_def_entries,
    };

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
        /// Build a minimal `ResolvedBounds` with the given thermal count,
        /// `n_stages`, and `k_max` (all other entity counts zero).
        ///
        /// `n_stages` must exceed the queried `stage_idx` so the anticipated
        /// layout the fixture builds places every plant inside the study horizon
        /// `[0, n_stages)`. `k_max` sizes the thermal-bounds stage axis: it must
        /// be large enough that every delivery stage `stage_idx + K_i` the test
        /// accesses falls within `[0, n_stages + k_max)` (callers whose
        /// deliveries all fall strictly inside `[0, n_stages)` pass `0`).
        /// `n_thermals` sizes the thermal slot axis: tests that read or mutate
        /// `thermal_bounds(t_idx, …)` for an active anticipated plant must size it
        /// to cover every `t_idx` they touch; pure state-out/row tests pass `0`.
        fn bounds_with_n_stages(
            n_stages: usize,
            k_max: usize,
            n_thermals: usize,
        ) -> ResolvedBounds {
            ResolvedBounds::new(
                &BoundsCountsSpec {
                    n_hydros: 0,
                    n_thermals,
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
                n_thermals,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated,
                k_max,
                anticipated_lead_stages,
                anticipated_thermal_indices: anticipated_thermal_indices
                    .into_iter()
                    .map(crate::indexer::ThermalSys::new)
                    .collect(),
                // Windowless: one `(None, None)` per plant, so the decision gate
                // reduces to the strict horizon clause. `study_stage_ids` lists the
                // study-stage ids so the in-range delivery lookup is safe.
                anticipated_windows: vec![(None, None); n_anticipated],
                anticipated_resolution: crate::lead_time::AnticipatedResolution::default(),
                study_stage_ids: (0..i32::try_from(self.bounds.n_stages()).unwrap_or(0)).collect(),
                has_penalty: false,
                // Sized to cover every active plant's delivery stage
                // (`stage_idx + K_i < n_stages`); `fill_anticipated_columns`
                // indexes these by delivery stage when pricing the decision column.
                cumulative_discount_factors: vec![1.0; self.bounds.n_stages() + k_max],
                total_hours_per_stage: vec![744.0; self.bounds.n_stages() + k_max],
                // No hydros ⇒ no filling targets.
                filling_v_target: BTreeMap::new(),
            }
        }
    }

    /// `fill_thermal_columns` skips the per-block objective for anticipated
    /// plants (leaving the `0.0` vec default) while still writing it for
    /// non-anticipated thermals — the order-independent replacement for the
    /// former write-then-zero pass. The anticipated plant's per-block **bounds**
    /// are still written.
    ///
    /// Fixture: two thermals, thermal 0 anticipated (`K=1`), thermal 1 standard,
    /// both with non-zero resolved `cost_per_mwh` so the skip is observable
    /// (a regression that priced anticipated plants would leave a non-zero
    /// objective on thermal 0).
    #[test]
    fn fill_thermal_columns_skips_objective_for_anticipated_plants() {
        use cobre_core::entities::thermal::AnticipatedConfig;
        use cobre_core::{EntityId, Thermal};

        const ANT_COST: f64 = 30.0;
        const STD_COST: f64 = 40.0;

        let thermals = vec![
            Thermal {
                id: EntityId(1),
                name: "T_ant".to_string(),
                operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                cost_per_mwh: ANT_COST,
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
                max_generation_mw: 100.0,
                cost_per_mwh: STD_COST,
                anticipated_config: None,
                entry_stage_id: None,
                exit_stage_id: None,
            },
        ];

        let mut fixtures = AntFixtures::new();
        // 10 stages, k_max=1; seed each thermal's resolved per-stage cost so the
        // objective write (when not skipped) is non-zero.
        fixtures.bounds = AntFixtures::bounds_with_n_stages(10, 1, 2);
        for stage in 0..10 {
            fixtures.bounds.thermal_bounds_mut(0, stage).cost_per_mwh = ANT_COST;
            fixtures
                .bounds
                .thermal_bounds_mut(0, stage)
                .max_generation_mw = 100.0;
            fixtures.bounds.thermal_bounds_mut(1, stage).cost_per_mwh = STD_COST;
            fixtures
                .bounds
                .thermal_bounds_mut(1, stage)
                .max_generation_mw = 100.0;
        }
        let mut ctx = fixtures.make_ctx(
            1,       // n_anticipated
            1,       // k_max
            vec![1], // anticipated_lead_stages: K_0 = 1
            vec![0], // anticipated_thermal_indices: thermal 0 is anticipated
            2,       // n_thermals
        );
        ctx.thermals = &thermals;

        let stage = two_block_stage(2, [372.0, 372.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 2);

        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut col_lower = vec![0.0_f64; layout.num_cols];
        let mut col_upper = vec![0.0_f64; layout.num_cols];
        let mut bufs = ColumnBufs {
            col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
        };

        fill_thermal_columns(&ctx, &stage, 2, &layout, &mut bufs);

        let n_blks = layout.n_blks;
        // Anticipated thermal 0: objective skipped (stays 0.0), bounds still set.
        // Thermal 0's block columns start at col_thermal_start (t_idx 0 offset).
        for blk in 0..n_blks {
            let col = layout.equipment.thermal.start + blk;
            assert_eq!(
                bufs.objective[col], 0.0,
                "anticipated thermal 0 objective must stay 0.0 at col {col}",
            );
            assert_eq!(
                bufs.col_upper[col], 100.0,
                "anticipated thermal 0 bounds must still be written at col {col}",
            );
        }
        // Standard thermal 1: objective priced as cost * block_hours.
        for blk in 0..n_blks {
            let col = layout.equipment.thermal.start + n_blks + blk;
            let expected = STD_COST * stage.blocks[blk].duration_hours;
            assert_eq!(
                bufs.objective[col], expected,
                "standard thermal 1 objective must be priced at col {col}",
            );
        }
    }

    /// Fills rows for all anticipated plants (always-active predicate).
    /// At stage 2 with `K_i=[1, 5]` and `n_anticipated=2`: both plants are active,
    /// so exactly two fishing rows are written.
    #[test]
    fn fishing_rows_fill_all_plants() {
        let mut fixtures = AntFixtures::new();
        fixtures.bounds = AntFixtures::bounds_with_n_stages(10, 0, 0);
        let ctx = fixtures.make_ctx(2, 5, vec![1, 5], vec![0, 1], 2);
        let stage = two_block_stage(2, [372.0, 372.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 2);

        // Always-active: both plants active at stage 2 → two fishing rows.
        assert_eq!(
            layout.anticipated.n_anticipated_fishing_rows, 2,
            "expected n_anticipated_fishing_rows == 2, got {}",
            layout.anticipated.n_anticipated_fishing_rows
        );

        let mut row_lower = vec![f64::NAN; layout.rows.num_rows];
        let mut row_upper = vec![f64::NAN; layout.rows.num_rows];

        fill_anticipated_fishing_rows(&layout, &mut row_lower, &mut row_upper);

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
        fixtures.bounds = AntFixtures::bounds_with_n_stages(10, 0, 0);
        let ctx = fixtures.make_ctx(2, 5, vec![1, 5], vec![0, 1], 2);
        let stage = two_block_stage(0, [372.0, 372.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        // Always-active: both plants are active at stage 0 → two fishing rows.
        assert_eq!(
            layout.anticipated.n_anticipated_fishing_rows, 2,
            "expected n_anticipated_fishing_rows == 2 at stage 0, got {}",
            layout.anticipated.n_anticipated_fishing_rows
        );

        let mut row_lower = vec![f64::NAN; layout.rows.num_rows];
        let mut row_upper = vec![f64::NAN; layout.rows.num_rows];

        fill_anticipated_fishing_rows(&layout, &mut row_lower, &mut row_upper);

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
        fill_anticipated_fishing_entries(&ctx, &stage, &layout, &mut col_entries);

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
    /// `fill_anticipated_columns`, `fill_anticipated_state_out_def_rows`,
    /// and `fill_anticipated_state_out_def_entries`.
    fn build_anticipated_ctx_n_stages_6() -> (AntFixtures, Stage) {
        let mut fixtures = AntFixtures::new();
        // Override bounds with a 6-stage, 2-thermal table; k_max = 3 matches the
        // anticipated config the state-out tests pass to `make_ctx`. The two
        // thermal slots back the active anticipated plants' delivery-stage bound
        // reads in `fill_anticipated_columns`; the row/def tests sharing this
        // fixture leave them unread.
        fixtures.bounds = AntFixtures::bounds_with_n_stages(6, 3, 2);
        let stage = two_block_stage(0, [372.0, 372.0]);
        (fixtures, stage)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Tests for the anticipated state-out columns (fill_anticipated_columns)
    // ─────────────────────────────────────────────────────────────────────────

    /// Active plants (`stage_idx + K_i < n_stages`) get `[-INF, +INF]` bounds;
    /// inactive plants (`stage_idx + K_i >= n_stages`) get `[0, 0]` bounds.
    ///
    /// Fixture: `n_anticipated=2`, `K=[2, 3]`, `n_stages=6`.
    /// Stage 0: both plants active  (0+2=2 < 6, 0+3=3 < 6) → `[-INF, +INF]`.
    /// Stage 5: both plants inactive (5+2=7 >= 6, 5+3=8 >= 6) → `[0, 0]`.
    ///
    /// Runs the full `fill_stage_columns` pipeline (not `fill_anticipated_columns`
    /// alone): a stage with NO genuine decision at all (stage 5's hypothetical
    /// delivery is out of horizon, so `genuine_decisions_at` is empty) has its
    /// ring slot frozen by `fill_anticipated_slot_columns`'s masking, not by
    /// `fill_anticipated_columns` — the two functions collaborate on the
    /// dormant-column convention exactly as `fill_stage_columns` composes them.
    #[test]
    fn test_fill_anticipated_columns_state_out_active_and_inactive() {
        let (fixtures, _) = build_anticipated_ctx_n_stages_6();
        let ctx = fixtures.make_ctx(
            2,          // n_anticipated
            3,          // k_max
            vec![2, 3], // anticipated_lead_stages: K=[2,3]
            vec![0, 1], // anticipated_thermal_indices
            0,          // n_thermals
        );

        // Stage 0: both plants active.
        let stage0 = two_block_stage(0, [372.0, 372.0]);
        let state = state_layout_for(&ctx);
        let layout0 = StageLayout::new(&ctx, &state, &stage0, 0);
        let (col_lower, col_upper, _objective) = fill_stage_columns(&ctx, &stage0, 0, &layout0);
        let leads = [2_usize, 3];
        for (i, &lead) in leads.iter().enumerate() {
            let col = layout0.anticipated.col_anticipated_slots_out_start + (lead - 1) * 2 + i;
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
        let stage5 = two_block_stage(5, [372.0, 372.0]);
        let state = state_layout_for(&ctx);
        let layout5 = StageLayout::new(&ctx, &state, &stage5, 5);
        assert_eq!(
            layout5.anticipated.n_anticipated_state_out_def_rows, 0,
            "stage 5 inactive: expected no def rows, got {}",
            layout5.anticipated.n_anticipated_state_out_def_rows,
        );
        let (col_lower5, col_upper5, _objective5) = fill_stage_columns(&ctx, &stage5, 5, &layout5);
        for (i, &lead) in leads.iter().enumerate() {
            let col = layout5.anticipated.col_anticipated_slots_out_start + (lead - 1) * 2 + i;
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
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        assert_eq!(
            layout.anticipated.n_anticipated_state_out_def_rows, 2,
            "expected n_anticipated_state_out_def_rows == 2, got {}",
            layout.anticipated.n_anticipated_state_out_def_rows
        );

        let mut row_lower = vec![f64::NEG_INFINITY; layout.rows.num_rows];
        let mut row_upper = vec![f64::INFINITY; layout.rows.num_rows];
        fill_anticipated_state_out_def_rows(&layout, &mut row_lower, &mut row_upper);

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
    /// - `(def_row_i, +1.0)` on plant `i`'s own newest ring slot
    /// - `(def_row_i, -1.0)` on `col_anticipated_decision_start + i`
    #[test]
    fn test_fill_anticipated_state_out_def_entries_two_active_plants() {
        let (fixtures, stage) = build_anticipated_ctx_n_stages_6();
        let ctx = fixtures.make_ctx(2, 3, vec![2, 3], vec![0, 1], 0);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_anticipated_state_out_def_entries(&ctx, 0, &layout, &mut col_entries);

        let leads = [2_usize, 3];
        for (k, &lead) in leads.iter().enumerate() {
            let row = layout.anticipated.row_anticipated_state_out_def_start + k;
            let col_state_out =
                layout.anticipated.col_anticipated_slots_out_start + (lead - 1) * 2 + k;
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
    // Two-sided masking: row cap and column freeze in one regression
    // ─────────────────────────────────────────────────────────────────────────

    /// An interior ring slot beyond the horizon-reachable cap gets BOTH sides of
    /// the masking contract together: no definition row AND a frozen `[0, 0]`
    /// outgoing column — never one without the other.
    ///
    /// Fixture: one plant, `K=3`, `n_stages=6`, evaluated at `stage_idx=4`
    /// (`horizon_cap = 6 - 4 - 1 = 1`). Interior slots are 0 and 1
    /// (`slot + 1 < K`); slot 0 is `< horizon_cap` (reachable), slot 1 is not
    /// (masked). Slot 2 is the plant's own newest slot, governed by the
    /// separate `anticipated_state_out_def` mechanism, not this one.
    #[test]
    fn anticipated_slot_masking_ships_row_cap_and_column_freeze_together() {
        let mut fixtures = AntFixtures::new();
        fixtures.bounds = AntFixtures::bounds_with_n_stages(6, 3, 1);
        let ctx = fixtures.make_ctx(1, 3, vec![3], vec![0], 1);
        let stage = two_block_stage(4, [372.0, 372.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 4);

        assert_eq!(
            layout.anticipated.anticipated_slot_row_pos,
            vec![Some(0), None, None],
            "slot 0 reachable (row pos 0), slot 1 masked, slot 2 belongs to the \
             deposit mechanism (never populated here)"
        );
        assert_eq!(layout.anticipated.n_anticipated_slot_definition_rows, 1);

        let (col_lower, col_upper, _objective) = fill_stage_columns(&ctx, &stage, 4, &layout);
        let (row_lower, row_upper) = fill_stage_rows(&ctx, &stage, 4, &layout);
        let col_entries = build_stage_matrix_entries(&ctx, &stage, 4, &layout);

        let base = layout.anticipated.col_anticipated_slots_out_start;
        let row_start = layout.anticipated.row_anticipated_slot_definition_start;

        // Reachable slot 0: free column, a defining row exists, and the CSC
        // carries the ring-shift's structural +1/-1 pair.
        let col0 = base;
        assert_eq!(
            col_lower[col0],
            f64::NEG_INFINITY,
            "slot 0 column must be free"
        );
        assert_eq!(col_upper[col0], f64::INFINITY, "slot 0 column must be free");
        let row0 = row_start;
        assert_eq!(
            row_lower[row0], 0.0,
            "slot 0 definition row must be an equality"
        );
        assert_eq!(
            row_upper[row0], 0.0,
            "slot 0 definition row must be an equality"
        );
        assert!(
            col_entries[col0]
                .iter()
                .any(|&(r, v)| r == row0 && (v - 1.0).abs() < 1e-15),
            "slot 0 outgoing column must carry the +1.0 structural term at its \
             definition row; got {:?}",
            col_entries[col0]
        );
        let incoming_slot1 = state.anticipated_state.start + 1;
        assert!(
            col_entries[incoming_slot1]
                .iter()
                .any(|&(r, v)| r == row0 && (v + 1.0).abs() < 1e-15),
            "slot 1's incoming pin must carry the -1.0 structural term at slot \
             0's definition row; got {:?}",
            col_entries[incoming_slot1]
        );

        // Masked slot 1: frozen column, no defining row, no CSC entry.
        let col1 = base + 1;
        assert_eq!(col_lower[col1], 0.0, "masked slot 1 column must be frozen");
        assert_eq!(col_upper[col1], 0.0, "masked slot 1 column must be frozen");
        assert!(
            col_entries[col1].is_empty(),
            "masked slot 1 must carry no CSC entries; got {:?}",
            col_entries[col1]
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Equivalence pin: ring-routed entries vs the open-coded reference formula
    // ─────────────────────────────────────────────────────────────────────────

    /// Equivalence pin: `fill_anticipated_slot_definition_entries`'s
    /// `DeliveryRing::emit_shift_rows` routing reproduces the open-coded reference
    /// formula (`+1` on `anticipated_slots_out.start + global_slot`, `-1` on
    /// `anticipated_state.start + (slot + 1) * n_anticipated + plant`, for every
    /// reachable global slot) on a fixed two-plant, heterogeneous-`K` fixture
    /// (`K = [3, 2]`).
    #[test]
    fn fill_anticipated_slot_definition_entries_matches_pre_migration_formula_across_heterogeneous_plants()
     {
        let (fixtures, stage) = build_anticipated_ctx_n_stages_6();
        let ctx = fixtures.make_ctx(2, 3, vec![3, 2], vec![0, 1], 2);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut actual: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_anticipated_slot_definition_entries(&layout, &mut actual);

        let n_ant = ctx.n_anticipated;
        let row_start = layout.anticipated.row_anticipated_slot_definition_start;
        let mut expected: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        let mut n_expected_reachable = 0_usize;
        for (global_slot, pos) in layout
            .anticipated
            .anticipated_slot_row_pos
            .iter()
            .enumerate()
        {
            let Some(pos) = *pos else { continue };
            let row = row_start + pos;
            let slot = global_slot / n_ant;
            let plant = global_slot % n_ant;
            expected[state.anticipated_slots_out.start + global_slot].push((row, 1.0));
            expected[state.anticipated_state.start + (slot + 1) * n_ant + plant].push((row, -1.0));
            n_expected_reachable += 1;
        }

        assert_eq!(
            n_expected_reachable, layout.anticipated.n_anticipated_slot_definition_rows,
            "fixture sanity: reachable count must match the layout's own count"
        );
        assert!(
            n_expected_reachable >= 3,
            "fixture must exercise multiple slots across both plants; got {n_expected_reachable}"
        );
        assert_eq!(
            actual, expected,
            "ring-routed entries must match the pre-migration open-coded formula"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // K = 0 exclusion
    // ─────────────────────────────────────────────────────────────────────────

    /// `K = 0` exclusion: a `LeadTime(720.0)` plant on the uniform 31-day-month
    /// calendar `[744,744,744,744]` h resolves `depth == [0,0,0,0]` (every
    /// delivery self-delivers, hand-derived from the calendar). No
    /// anticipated slot, deposit row, interior-shift row, or fishing row is
    /// ever emitted for this plant, at any stage — the stage LP dispatches its
    /// generation as ordinary, unconstrained thermal output (no fishing
    /// coupling), never an underflow.
    #[test]
    fn k0_sub_stage_lead_emits_no_anticipated_rows_or_fishing_coupling() {
        let mut fixtures = AntFixtures::new();
        fixtures.bounds = AntFixtures::bounds_with_n_stages(4, 0, 1);
        let ctx = fixtures.make_ctx(1, 0, vec![0], vec![0], 1);

        let mut state = state_layout_for(&ctx);
        state.set_anticipated_resolution(crate::lead_time::AnticipatedResolution::resolve(
            &[crate::lead_time::LeadTime::Time(720.0)],
            &[744.0, 744.0, 744.0, 744.0],
            4,
        ));

        for stage_idx in 0..4 {
            let stage = two_block_stage(stage_idx, [372.0, 372.0]);
            let layout = StageLayout::new(&ctx, &state, &stage, stage_idx);
            assert_eq!(
                layout.anticipated.n_anticipated_fishing_rows, 0,
                "stage {stage_idx}: K=0 must exclude the fishing row entirely"
            );
            assert_eq!(
                layout.anticipated.n_anticipated_state_out_def_rows, 0,
                "stage {stage_idx}: K=0 must exclude the deposit row entirely"
            );
            assert_eq!(
                layout.anticipated.n_anticipated_slot_definition_rows, 0,
                "stage {stage_idx}: K=0 must exclude every interior-shift row"
            );
            assert_eq!(
                layout.anticipated_decision().len(),
                ctx.n_anticipated,
                "stage {stage_idx}: the decision-column block stays uniformly \
                 n_anticipated wide even when every plant is K=0 (all rows excluded, \
                 the columns are not)"
            );

            let mut row_lower = vec![f64::NAN; layout.rows.num_rows];
            let mut row_upper = vec![f64::NAN; layout.rows.num_rows];
            fill_anticipated_fishing_rows(&layout, &mut row_lower, &mut row_upper);
            fill_anticipated_state_out_def_rows(&layout, &mut row_lower, &mut row_upper);

            let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
            fill_anticipated_fishing_entries(&ctx, &stage, &layout, &mut col_entries);
            fill_anticipated_state_out_def_entries(&ctx, stage_idx, &layout, &mut col_entries);

            // The plant's ordinary thermal generation columns carry no entry
            // at all from either anticipated row family — unconstrained by
            // any fishing coupling.
            for blk in 0..layout.n_blks {
                let col_gen = layout
                    .block_grid()
                    .flat(layout.equipment.thermal.start, 0, blk);
                assert!(
                    col_entries[col_gen].is_empty(),
                    "stage {stage_idx} blk {blk}: thermal generation column must carry no \
                     anticipated coupling entry; got {:?}",
                    col_entries[col_gen]
                );
            }
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
            2,          // n_thermals: must cover thermal indices 0 and 1 so the
                        // fishing-row entry resolves to a real thermal column.
        );

        let state = state_layout_for(&ctx);

        let layout = StageLayout::new(&ctx, &state, &stage, 0);
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
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::similar_names,
    clippy::too_many_lines
)]
mod pumping_water_tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::{
        BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties, CascadeTopology, CoefficientRef,
        ConstraintExpression, ConstraintSense, ContractStageBounds, ContractType, DeficitSegment,
        EnergyContract, EntityId, GenericConstraint, Hydro, HydroGenerationModel, HydroStageBounds,
        HydroStagePenalties, Line, LineStageBounds, LineStagePenalties, LinearTerm,
        NcsStagePenalties, PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds,
        PumpingStation, ResolvedBounds, ResolvedExchangeFactors, ResolvedGenericConstraintBounds,
        ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties, SlackConfig,
        Stage, Thermal, ThermalStageBounds, VariableRef,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::{EvaporationModel, EvaporationModelSet, ProductionModelSet};
    use crate::indexer::{EvapLocal, HydroSys, LineSys, StateDim, StateLayout};
    use crate::lead_time::{SpreadResolution, resolve_spread};
    use crate::resolved_parameters::ResolvedParameters;

    use super::super::M3S_TO_HM3;
    use super::super::columns::{ColumnBufs, fill_pumping_columns};
    use super::super::layout::{ResolvedTables, StageLayout, TemplateBuildCtx};
    use super::super::test_support::{state_layout_for, two_block_stage, zero_hydro_penalties};
    use super::{
        LpMatrixBuffers, assemble_csc, build_stage_matrix_entries, fill_generic_constraint_entries,
        fill_load_balance_entries, fill_pumping_water_entries,
        fill_transit_bucket_definition_entries, resolve_chrono_arrival_density,
    };

    const N_STAGES: usize = 1;

    /// Minimal independent (no-downstream) constant-productivity hydro.
    fn fixture_hydro(id: i32) -> Hydro {
        fixture_hydro_ds(id, None)
    }

    /// `fixture_hydro` with a caller-chosen `downstream_id`, so a two-reservoir
    /// cascade can be built from the fixture hydros. `fixture_hydro` delegates
    /// here with `None`, keeping every existing caller's cascade empty.
    fn fixture_hydro_ds(id: i32, downstream_id: Option<i32>) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: downstream_id.map(EntityId),
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
        }
    }

    /// A single bus with one unbounded deficit segment, on `EntityId(1)` (the bus
    /// the fixture hydros and `station` helper reference).
    fn fixture_bus(id: i32) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("B{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
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
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
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

    /// A bus with a caller-chosen deficit-segment count and excess cost so the
    /// multi-entity permutation test can give each bus a distinct CSC footprint:
    /// `fill_load_balance_entries` emits one deficit column per
    /// `deficit_segments.len()`, so distinct segment counts make the deficit
    /// column block bus-position-dependent — a bus permutation that escaped the
    /// ID-sort would land the wrong segment count in the wrong column block.
    fn fixture_bus_with(id: i32, n_segments: usize, excess_cost: f64) -> Bus {
        Bus {
            id: EntityId(id),
            name: format!("B{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: (0..n_segments)
                .map(|_| DeficitSegment {
                    depth_mw: None,
                    cost_per_mwh: 1000.0,
                })
                .collect(),
            excess_cost,
        }
    }

    /// A thermal plant on `bus_id` with distinct generation bounds and cost so a
    /// permutation that mislabelled which thermal owns which column is observable.
    fn fixture_thermal(id: i32, bus_id: i32, min_gen: f64, max_gen: f64, cost: f64) -> Thermal {
        Thermal {
            id: EntityId(id),
            name: format!("T{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(bus_id),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: cost,
            min_generation_mw: min_gen,
            max_generation_mw: max_gen,
            anticipated_config: None,
        }
    }

    /// A transmission line between `source_bus`/`target_bus` with distinct
    /// capacities. Distinct bus pairs across lines make the load-balance fill
    /// (±1.0 into source/target bus rows) line-position- and bus-position-
    /// dependent, so a permutation that escaped either ID-sort changes the CSC.
    fn fixture_line(id: i32, source_bus: i32, target_bus: i32, direct: f64, reverse: f64) -> Line {
        Line {
            id: EntityId(id),
            name: format!("L{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            source_bus_id: EntityId(source_bus),
            target_bus_id: EntityId(target_bus),
            entry_stage_id: None,
            exit_stage_id: None,
            direct_capacity_mw: direct,
            reverse_capacity_mw: reverse,
            losses_percent: 0.0,
            exchange_cost: 0.0,
        }
    }

    /// Owns the data backing a two-hydro `TemplateBuildCtx` carrying pumping
    /// stations, thermals, and lines. Every entity slice is stored in canonical
    /// (ID-sorted) order; the position maps are derived from those sorted slices,
    /// exactly as `SystemBuilder::build` produces them in production. The ID-sort
    /// is what makes two declaration orders of the same entities converge to one
    /// ctx — the property the permutation tests assert.
    struct PumpFixtures {
        hydros: Vec<Hydro>,
        stations: Vec<PumpingStation>,
        buses: Vec<Bus>,
        thermals: Vec<Thermal>,
        lines: Vec<Line>,
        /// Energy contracts, id-sorted; empty by default. The contract
        /// load-balance-sign tests supply import/export contracts here.
        contracts: Vec<EnergyContract>,
        hydro_pos: BTreeMap<EntityId, usize>,
        pumping_pos: BTreeMap<EntityId, usize>,
        bus_pos: BTreeMap<EntityId, usize>,
        thermal_pos: BTreeMap<EntityId, usize>,
        line_pos: BTreeMap<EntityId, usize>,
        contract_pos: BTreeMap<EntityId, usize>,
        n_contract_import: usize,
        n_contract_export: usize,
        par_lp: PrecomputedPar,
        /// AR order the ctx exposes as `max_par_order`. Zero by default (no
        /// inflow-lag columns reserved); raised by [`PumpFixtures::with_par_lp`]
        /// to match the injected `par_lp` so the AR-lag water term can fire.
        max_par_order: usize,
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
        /// declaration order, with no thermals or lines. Delegates to
        /// [`PumpFixtures::new_full`], which sorts every slice by `id.0` (the
        /// canonical operation `SystemBuilder::build` performs) before deriving
        /// position maps, so the resulting ctx is declaration-order-invariant.
        fn new_with_buses(
            hydros: Vec<Hydro>,
            stations: Vec<PumpingStation>,
            buses: Vec<Bus>,
        ) -> Self {
            Self::new_full(hydros, stations, buses, Vec::new(), Vec::new())
        }

        /// Build a fixture with buses and energy contracts (no thermals/lines),
        /// for the contract load-balance-sign tests.
        fn new_with_contracts(
            hydros: Vec<Hydro>,
            buses: Vec<Bus>,
            contracts: Vec<EnergyContract>,
        ) -> Self {
            Self::new_full_with_contracts(
                hydros,
                Vec::new(),
                buses,
                Vec::new(),
                Vec::new(),
                contracts,
            )
        }

        /// Build a fixture carrying hydros, stations, buses, thermals, and lines in
        /// arbitrary declaration order. Every slice is sorted by `id.0` before
        /// deriving its position map and bounds table, mirroring
        /// `SystemBuilder::build`; this canonicalisation is what makes two
        /// declaration orders of the same entities converge to one ctx (the
        /// invariant the permutation tests assert). Thermal and line bounds are
        /// written per-entity from the sorted slices so a column/bound mislabel
        /// under permutation is observable.
        fn new_full(
            hydros: Vec<Hydro>,
            stations: Vec<PumpingStation>,
            buses: Vec<Bus>,
            thermals: Vec<Thermal>,
            lines: Vec<Line>,
        ) -> Self {
            Self::new_full_with_contracts(hydros, stations, buses, thermals, lines, Vec::new())
        }

        #[allow(clippy::too_many_lines)]
        fn new_full_with_contracts(
            mut hydros: Vec<Hydro>,
            mut stations: Vec<PumpingStation>,
            mut buses: Vec<Bus>,
            mut thermals: Vec<Thermal>,
            mut lines: Vec<Line>,
            mut contracts: Vec<EnergyContract>,
        ) -> Self {
            hydros.sort_by_key(|h| h.id.0);
            stations.sort_by_key(|s| s.id.0);
            buses.sort_by_key(|b| b.id.0);
            thermals.sort_by_key(|t| t.id.0);
            lines.sort_by_key(|l| l.id.0);
            contracts.sort_by_key(|c| c.id.0);

            let hydro_pos: BTreeMap<EntityId, usize> =
                hydros.iter().enumerate().map(|(i, h)| (h.id, i)).collect();
            let pumping_pos: BTreeMap<EntityId, usize> = stations
                .iter()
                .enumerate()
                .map(|(i, s)| (s.id, i))
                .collect();
            let bus_pos: BTreeMap<EntityId, usize> =
                buses.iter().enumerate().map(|(i, b)| (b.id, i)).collect();
            let thermal_pos: BTreeMap<EntityId, usize> = thermals
                .iter()
                .enumerate()
                .map(|(i, t)| (t.id, i))
                .collect();
            let line_pos: BTreeMap<EntityId, usize> =
                lines.iter().enumerate().map(|(i, l)| (l.id, i)).collect();
            let contract_pos: BTreeMap<EntityId, usize> = contracts
                .iter()
                .enumerate()
                .map(|(i, c)| (c.id, i))
                .collect();
            let n_contract_import = contracts
                .iter()
                .filter(|c| c.contract_type == ContractType::Import)
                .count();
            let n_contract_export = contracts
                .iter()
                .filter(|c| c.contract_type == ContractType::Export)
                .count();

            let mut bounds = ResolvedBounds::new(
                &BoundsCountsSpec {
                    n_hydros: hydros.len(),
                    n_thermals: thermals.len(),
                    n_lines: lines.len(),
                    n_pumping: stations.len(),
                    n_contracts: contracts.len(),
                    n_stages: N_STAGES,
                    k_max: 0,
                },
                &default_bounds_defaults(),
            );
            // Distinct per-contract bounds from the sorted slice so a column/bound
            // mislabel under permutation is observable.
            for (c_idx, c) in contracts.iter().enumerate() {
                for stage_idx in 0..N_STAGES {
                    *bounds.contract_bounds_mut(c_idx, stage_idx) = ContractStageBounds {
                        min_mw: c.min_mw,
                        max_mw: c.max_mw,
                        price_per_mwh: c.price_per_mwh,
                    };
                }
            }
            // Distinct per-station bounds so a column/bound mismatch is observable.
            for (p_idx, s) in stations.iter().enumerate() {
                for stage_idx in 0..N_STAGES {
                    *bounds.pumping_bounds_mut(p_idx, stage_idx) = PumpingStageBounds {
                        min_flow_m3s: s.min_flow_m3s,
                        max_flow_m3s: s.max_flow_m3s,
                    };
                }
            }
            // Distinct per-thermal generation bounds and cost, taken from the
            // sorted slice, so a permutation that mislabelled which thermal owns
            // which column would change the resolved bounds.
            for (t_idx, t) in thermals.iter().enumerate() {
                for stage_idx in 0..N_STAGES {
                    *bounds.thermal_bounds_mut(t_idx, stage_idx) = ThermalStageBounds {
                        min_generation_mw: t.min_generation_mw,
                        max_generation_mw: t.max_generation_mw,
                        cost_per_mwh: t.cost_per_mwh,
                    };
                }
            }
            // Distinct per-line capacities from the sorted slice, same rationale.
            for (l_idx, l) in lines.iter().enumerate() {
                for stage_idx in 0..N_STAGES {
                    *bounds.line_bounds_mut(l_idx, stage_idx) = LineStageBounds {
                        direct_mw: l.direct_capacity_mw,
                        reverse_mw: l.reverse_capacity_mw,
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

            // Built from the sorted hydros so a fixture carrying `downstream_id`
            // produces a real cascade; output-neutral for the `fixture_hydro`
            // callers, whose `downstream_id: None` yields the same empty cascade
            // `CascadeTopology::build(&[])` did.
            let cascade = CascadeTopology::build(&hydros);

            Self {
                hydros,
                stations,
                buses,
                thermals,
                lines,
                contracts,
                hydro_pos,
                pumping_pos,
                bus_pos,
                thermal_pos,
                line_pos,
                contract_pos,
                n_contract_import,
                n_contract_export,
                par_lp: PrecomputedPar::default(),
                cascade,
                max_par_order: 0,
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

        /// Inject a `PrecomputedPar` and align `max_par_order` to its order, so
        /// the inflow-lag columns are reserved and the `−ζ·ψ` AR-lag water term
        /// fires. The default fixture carries `PrecomputedPar::default()` (no
        /// `psi`, `max_par_order: 0`), under which the AR-lag term is dormant.
        fn with_par_lp(mut self, par_lp: PrecomputedPar) -> Self {
            self.max_par_order = par_lp.max_order();
            self.par_lp = par_lp;
            self
        }

        /// Replace the default empty penalty table with a properly-sized all-zero
        /// (plus a small spillage cost) `ResolvedPenalties` so the FULL template
        /// build (`build_single_stage_template`, which reads per-hydro penalties)
        /// does not index an empty table. The matrix-only tests use
        /// `ResolvedPenalties::empty()` because `build_stage_matrix_entries` never
        /// reads penalties; the solver-backed duals test needs this.
        fn with_resolved_penalties(mut self) -> Self {
            let hydro = HydroStagePenalties {
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
                inflow_nonnegativity_cost: 0.0,
            };
            self.penalties = ResolvedPenalties::new(
                &PenaltiesCountsSpec {
                    n_hydros: self.hydros.len(),
                    n_buses: self.buses.len(),
                    n_lines: self.lines.len(),
                    n_ncs: 0,
                    n_stages: N_STAGES,
                },
                &PenaltiesDefaults {
                    hydro,
                    bus: BusStagePenalties { excess_cost: 0.0 },
                    line: LineStagePenalties { exchange_cost: 0.0 },
                    ncs: NcsStagePenalties {
                        curtailment_cost: 0.0,
                    },
                },
            );
            self
        }

        /// Size a `ResolvedPenalties` carrying nonzero directional evaporation
        /// violation costs so the per-block `f_evap_plus`/`f_evap_minus` slack
        /// objectives are observable in the column build.
        fn with_evap_penalties(mut self, neg_cost: f64, pos_cost: f64) -> Self {
            let hydro = HydroStagePenalties {
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
                evaporation_violation_pos_cost: pos_cost,
                evaporation_violation_neg_cost: neg_cost,
                inflow_nonnegativity_cost: 0.0,
            };
            self.penalties = ResolvedPenalties::new(
                &PenaltiesCountsSpec {
                    n_hydros: self.hydros.len(),
                    n_buses: self.buses.len(),
                    n_lines: self.lines.len(),
                    n_ncs: 0,
                    n_stages: N_STAGES,
                },
                &PenaltiesDefaults {
                    hydro,
                    bus: BusStagePenalties { excess_cost: 0.0 },
                    line: LineStagePenalties { exchange_cost: 0.0 },
                    ncs: NcsStagePenalties {
                        curtailment_cost: 0.0,
                    },
                },
            );
            self
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            TemplateBuildCtx {
                hydros: &self.hydros,
                thermals: &self.thermals,
                lines: &self.lines,
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
                thermal_pos: self.thermal_pos.clone(),
                line_pos: self.line_pos.clone(),
                bus_pos: self.bus_pos.clone(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &self.generic_constraints,
                non_controllable_sources: &[],
                pumping_stations: &self.stations,
                pumping_pos: self.pumping_pos.clone(),
                n_pumping: self.stations.len(),
                contracts: &self.contracts,
                contract_pos: self.contract_pos.clone(),
                n_contract_import: self.n_contract_import,
                n_contract_export: self.n_contract_export,
                diversion_upstream: HashMap::new(),
                arc_stage_weights: HashMap::new(),
                arc_spread_chrono: HashMap::new(),
                arc_arrival_density: HashMap::new(),
                per_stage_mask: Vec::new(),
                n_hydros: self.hydros.len(),
                n_thermals: self.thermals.len(),
                n_lines: self.lines.len(),
                n_buses: self.buses.len(),
                max_par_order: self.max_par_order,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                anticipated_windows: vec![],
                anticipated_resolution: crate::lead_time::AnticipatedResolution::default(),
                study_stage_ids: vec![],
                has_penalty: false,
                cumulative_discount_factors: vec![1.0; N_STAGES],
                total_hours_per_stage: vec![744.0; N_STAGES],
                // These single-stage fixtures decouple `stage.id` from
                // `stage_idx` (every phase is exercised at `stage_idx = 0` against
                // one bounds row), so the filling window's stage ids all resolve to
                // idx 0. The backward fold reads `total_hours_per_stage[0]` and
                // `hydro_bounds(h, 0)` for every filling stage, matching how each
                // stage is built. Covers ids 0..=8 — wider than any filling window
                // under test (max entry = 4).
                filling_v_target: super::super::template::build_filling_v_target(
                    &self.hydros,
                    &self.bounds,
                    &[744.0; N_STAGES],
                    &(0..=8_i32).map(|id| (id, 0_usize)).collect(),
                ),
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
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

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

        fill_pumping_columns(&ctx, &stage, 0, &layout, &mut bufs);

        let n_blks = layout.n_blks;
        for (p_idx, s) in ctx.pumping_stations.iter().enumerate() {
            for blk in 0..n_blks {
                let col = layout.equipment.col_pumping_start + p_idx * n_blks + blk;
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
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_pumping_water_entries(&ctx, &stage, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        let source_pos = ctx.hydro_pos[&EntityId(1)];
        let dest_pos = ctx.hydro_pos[&EntityId(2)];
        let row_source = layout.rows.water_balance.start + source_pos;
        let row_dest = layout.rows.water_balance.start + dest_pos;

        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col = layout.equipment.col_pumping_start + blk;
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
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_pumping_water_entries(&ctx, &stage, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        let dest_pos = ctx.hydro_pos[&EntityId(2)];
        let row_dest = layout.rows.water_balance.start + dest_pos;
        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col = layout.equipment.col_pumping_start + blk;
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
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_pumping_water_entries(&ctx, &stage, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        let source_pos = ctx.hydro_pos[&EntityId(1)];
        let row_source = layout.rows.water_balance.start + source_pos;
        for blk in 0..n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            let col = layout.equipment.col_pumping_start + blk;
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
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        let b_idx = ctx.bus_pos[&EntityId(1)];
        for blk in 0..n_blks {
            let row = layout.rows.load_balance.start + b_idx * n_blks + blk;
            let col = layout.equipment.col_pumping_start + blk;
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

    /// An import-only `EnergyContract` on `bus_id` carrying `id`/`bus_id`/
    /// `contract_type`; every other field inert.
    fn contract(id: i32, bus_id: i32, contract_type: ContractType) -> EnergyContract {
        EnergyContract {
            id: EntityId(id),
            name: format!("C{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(bus_id),
            contract_type,
            entry_stage_id: None,
            exit_stage_id: None,
            price_per_mwh: 0.0,
            min_mw: 0.0,
            max_mw: 100.0,
        }
    }

    /// An import contract enters its bus load-balance row with `+1.0` per block —
    /// injection into the bus. This sign is independent of the price sign.
    #[test]
    fn contract_import_enters_bus_row_with_plus_one() {
        let fixtures = PumpFixtures::new_with_contracts(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![fixture_bus(1)],
            vec![contract(10, 1, ContractType::Import)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        let b_idx = ctx.bus_pos[&EntityId(1)];
        for blk in 0..n_blks {
            let row = layout.rows.load_balance.start + b_idx * n_blks + blk;
            let col = layout.equipment.col_contract_import_start + blk;
            assert_eq!(
                col_entries[col],
                vec![(row, 1.0)],
                "blk {blk}: import contract column must carry only (row, +1.0)"
            );
        }
    }

    /// An export contract enters its bus load-balance row with `-1.0` per block —
    /// withdrawal from the bus. Flipping this would make an export feed the bus.
    #[test]
    fn contract_export_enters_bus_row_with_minus_one() {
        let fixtures = PumpFixtures::new_with_contracts(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![fixture_bus(1)],
            vec![contract(10, 1, ContractType::Export)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        let b_idx = ctx.bus_pos[&EntityId(1)];
        for blk in 0..n_blks {
            let row = layout.rows.load_balance.start + b_idx * n_blks + blk;
            let col = layout.equipment.col_contract_export_start + blk;
            assert_eq!(
                col_entries[col],
                vec![(row, -1.0)],
                "blk {blk}: export contract column must carry only (row, -1.0)"
            );
        }
    }

    /// Mixed import/export at distinct per-family slots land on the right column
    /// bases with the right signs: import at `col_contract_import_start`, export at
    /// `col_contract_export_start`.
    #[test]
    fn contract_mixed_import_export_use_per_family_bases() {
        let fixtures = PumpFixtures::new_with_contracts(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![fixture_bus(1)],
            vec![
                contract(10, 1, ContractType::Import),
                contract(20, 1, ContractType::Export),
            ],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        let b_idx = ctx.bus_pos[&EntityId(1)];
        for blk in 0..n_blks {
            let row = layout.rows.load_balance.start + b_idx * n_blks + blk;
            let import_col = layout.equipment.col_contract_import_start + blk;
            let export_col = layout.equipment.col_contract_export_start + blk;
            assert_eq!(
                col_entries[import_col],
                vec![(row, 1.0)],
                "blk {blk}: import"
            );
            assert_eq!(
                col_entries[export_col],
                vec![(row, -1.0)],
                "blk {blk}: export"
            );
        }
    }

    /// Two imports on one bus address distinct per-family slots: the first
    /// (`family_slot` 0) lands on `col_contract_import_start + 0*n_blks + blk`, the
    /// second (`family_slot` 1) on `col_contract_import_start + 1*n_blks + blk`. A
    /// regression to using `c_sys` instead of `family_slot` would collide them.
    #[test]
    fn contract_second_import_uses_family_slot_one() {
        let fixtures = PumpFixtures::new_with_contracts(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![fixture_bus(1)],
            vec![
                contract(10, 1, ContractType::Import),
                contract(20, 1, ContractType::Import),
            ],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        let b_idx = ctx.bus_pos[&EntityId(1)];
        for blk in 0..n_blks {
            let row = layout.rows.load_balance.start + b_idx * n_blks + blk;
            let slot0_col = layout.equipment.col_contract_import_start + blk;
            let slot1_col = layout.equipment.col_contract_import_start + n_blks + blk;
            assert_eq!(
                col_entries[slot0_col],
                vec![(row, 1.0)],
                "blk {blk}: first import (family_slot 0)"
            );
            assert_eq!(
                col_entries[slot1_col],
                vec![(row, 1.0)],
                "blk {blk}: second import (family_slot 1)"
            );
        }
    }

    /// A contract whose `bus_id` is absent from `bus_pos` writes no load-balance
    /// entry and does not panic.
    #[test]
    fn contract_missing_bus_skips_without_panic() {
        let fixtures = PumpFixtures::new_with_contracts(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![fixture_bus(1)],
            vec![contract(10, 99, ContractType::Import)],
        );
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        for blk in 0..n_blks {
            let col = layout.equipment.col_contract_import_start + blk;
            assert!(
                col_entries[col].is_empty(),
                "blk {blk}: contract on an unmapped bus must write no load-balance entry"
            );
        }
    }

    /// AC: a contracts-bearing system with a generic constraint referencing
    /// `ContractImport` assembles its full stage matrix in a debug build with no
    /// `debug_assert!` firing — the mandatory tripwire's terminal condition.
    #[test]
    fn contracts_with_generic_constraint_assemble_without_debug_assert() {
        let import_term = LinearTerm {
            variable: VariableRef::ContractImport {
                contract_id: EntityId(10),
                block_id: None,
            },
            coefficient: CoefficientRef::Literal(1.0),
            scale: 1.0,
        };
        let constraint = GenericConstraint {
            id: EntityId(1),
            name: "c-import-cap".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![import_term],
            },
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        };
        let fixtures = PumpFixtures::new_with_contracts(
            vec![fixture_hydro(1), fixture_hydro(2)],
            vec![fixture_bus(1)],
            vec![contract(10, 1, ContractType::Import)],
        )
        .with_generic_constraint(constraint, 50.0);
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        // Exercises fill_load_balance_entries (contract block) and
        // fill_generic_constraint_entries (the resolved ContractImport arm). Under
        // debug_assertions every debug_assert! is live; reaching the assertion below
        // proves none fired.
        let col_entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
        assert_eq!(col_entries.len(), layout.num_cols);
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
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);

        let n_blks = layout.n_blks;
        for blk in 0..n_blks {
            let col = layout.equipment.col_pumping_start + blk;
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
            let stage = two_block_stage(0, [300.0, 444.0]);
            let state = state_layout_for(&ctx);
            let layout = StageLayout::new(&ctx, &state, &stage, 0);
            let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
            fill_load_balance_entries(&ctx, 0, &layout, &mut col_entries);
            // Truncate to the non-pumping column region: with zero stations the
            // layout has no pumping columns, so compare the shared prefix that both
            // layouts share (generation/thermal/line/deficit/excess columns are all
            // indexed before the pumping block).
            (layout.equipment.col_pumping_start, col_entries)
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
            let stage = two_block_stage(0, [300.0, 444.0]);
            let state = state_layout_for(&ctx);
            let layout = StageLayout::new(&ctx, &state, &stage, 0);
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

    /// Build the full stage CSC for a multi-entity system (2 hydros, 3 buses, 2
    /// thermals on different buses, 2 lines on different bus pairs, 1 generic
    /// constraint over a thermal/line/bus triple) twice with EVERY entity family's
    /// declaration order reversed between the two builds, and assert the assembled
    /// CSC arrays are byte-identical. This is the determinism backstop for the
    /// bus/line/thermal/generic-constraint families the single-bus
    /// `csc_byte_identical_under_permuted_declaration_order` cannot scramble: a
    /// map-iterating fill that reintroduced declaration-order dependence on any of
    /// these families would diverge here.
    ///
    /// The constraint references a `ThermalGeneration`, a `LineExchange`, and a
    /// `BusDeficit` term so the resolver path through
    /// `thermal_pos`/`line_pos`/`bus_pos` participates in the asserted CSC, not
    /// only the load-balance fill. Distinct per-entity attributes (thermals on
    /// different buses with distinct bounds, lines on different bus pairs with
    /// distinct capacities, buses with distinct deficit-segment counts) make the
    /// assertion load-bearing: a permutation that mislabelled which entity owns
    /// which slot would change the CSC.
    #[test]
    fn csc_byte_identical_under_permuted_multi_entity_order() {
        // Generic constraint id 7:
        //   thermal_gen(10) + line_exchange(100) + bus_deficit(2) <= 100
        // All referenced ids ARE present in the position maps, so the resolver
        // contributes real entries rather than silently returning an empty vec.
        let make_constraint = || GenericConstraint {
            id: EntityId(7),
            name: "gc_multi".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![
                    LinearTerm {
                        coefficient: CoefficientRef::Literal(1.0),
                        scale: 1.0,
                        variable: VariableRef::ThermalGeneration {
                            thermal_id: EntityId(10),
                            block_id: None,
                        },
                    },
                    LinearTerm {
                        coefficient: CoefficientRef::Literal(1.0),
                        scale: 1.0,
                        variable: VariableRef::LineExchange {
                            line_id: EntityId(100),
                            block_id: None,
                        },
                    },
                    LinearTerm {
                        coefficient: CoefficientRef::Literal(1.0),
                        scale: 1.0,
                        variable: VariableRef::BusDeficit {
                            bus_id: EntityId(2),
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

        // Run the production assemble sequence into a CSC for a fixture. Takes the
        // fixture by reference so the caller can keep it alive and build a layout
        // from it for the offset reads — the per-call `StageLayout` borrows the
        // function-local ctx/state and cannot escape the closure.
        let assemble = |fixtures: &PumpFixtures| {
            let ctx = fixtures.make_ctx();
            let stage = two_block_stage(0, [300.0, 444.0]);
            let state = state_layout_for(&ctx);
            let layout = StageLayout::new(&ctx, &state, &stage, 0);

            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            let mut col_upper = vec![f64::INFINITY; layout.num_cols];
            let mut objective = vec![0.0_f64; layout.num_cols];
            let mut row_lower = vec![f64::NEG_INFINITY; layout.rows.num_rows];
            let mut row_upper = vec![f64::INFINITY; layout.rows.num_rows];
            let mut buffers = LpMatrixBuffers {
                col_entries: &mut entries,
                col_upper: &mut col_upper,
                objective: &mut objective,
                row_lower: &mut row_lower,
                row_upper: &mut row_upper,
            };
            fill_generic_constraint_entries(&ctx, &stage, 0, &layout, &mut buffers);

            // Mirror the production per-column row-sort (see
            // build_single_stage_template) before assembling the CSC.
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };

        // Order A: every family declared ascending. The fixture is kept alive so
        // the order-A layout (for the offset reads below) is built from it.
        let fixtures_a = PumpFixtures::new_full(
            vec![fixture_hydro(1), fixture_hydro(2)],
            Vec::new(),
            vec![
                fixture_bus_with(1, 1, 1.0),
                fixture_bus_with(2, 2, 3.0),
                fixture_bus_with(3, 1, 5.0),
            ],
            vec![
                fixture_thermal(10, 1, 0.0, 30.0, 12.0),
                fixture_thermal(20, 3, 5.0, 45.0, 27.0),
            ],
            vec![
                fixture_line(100, 1, 2, 50.0, 40.0),
                fixture_line(200, 2, 3, 70.0, 60.0),
            ],
        )
        .with_generic_constraint(make_constraint(), 100.0);
        let csc_a = assemble(&fixtures_a);

        // Order B: the identical entities, every family declared in reverse.
        let fixtures_b = PumpFixtures::new_full(
            vec![fixture_hydro(2), fixture_hydro(1)],
            Vec::new(),
            vec![
                fixture_bus_with(3, 1, 5.0),
                fixture_bus_with(2, 2, 3.0),
                fixture_bus_with(1, 1, 1.0),
            ],
            vec![
                fixture_thermal(20, 3, 5.0, 45.0, 27.0),
                fixture_thermal(10, 1, 0.0, 30.0, 12.0),
            ],
            vec![
                fixture_line(200, 2, 3, 70.0, 60.0),
                fixture_line(100, 1, 2, 50.0, 40.0),
            ],
        )
        .with_generic_constraint(make_constraint(), 100.0);
        let csc_b = assemble(&fixtures_b);

        // Order-A layout (held by the test, owning its ctx/state) for the offset
        // reads below. The layout offsets are declaration-order-invariant, so this
        // matches the layout order A's CSC was assembled with.
        let ctx_a = fixtures_a.make_ctx();
        let stage_a = two_block_stage(0, [300.0, 444.0]);
        let state_a = state_layout_for(&ctx_a);
        let layout_a = StageLayout::new(&ctx_a, &state_a, &stage_a, 0);

        assert_eq!(csc_a.0, csc_b.0, "col_starts must be byte-identical");
        assert_eq!(csc_a.1, csc_b.1, "row_indices must be byte-identical");
        assert_eq!(csc_a.2, csc_b.2, "values must be byte-identical");

        // Criterion 3: prove the generic-constraint resolver path ran (not just
        // the load-balance fill) by reading the generic row's coefficients on the
        // resolved thermal/line/deficit columns from order A's CSC. The expression
        // is block-dependent (ThermalGeneration/LineExchange/BusDeficit), so it
        // expands to one generic row per block; probe every block.
        let n_blks = layout_a.n_blks;
        assert_eq!(
            layout_a.rows.n_generic_rows, n_blks,
            "block-dependent generic constraint must expand to one row per block"
        );
        let grid = layout_a.block_grid();
        let t_pos = 0; // thermal id 10 sorts to position 0.
        let l_pos = 0; // line id 100 sorts to position 0.
        let b_pos = 1; // bus id 2 sorts to position 1 (buses 1,2,3).
        for blk in 0..n_blks {
            let row = i32::try_from(layout_a.rows.row_generic_start + blk).unwrap();
            let coeff_at = |col: usize| -> f64 {
                let start = usize::try_from(csc_a.0[col]).unwrap();
                let end = usize::try_from(csc_a.0[col + 1]).unwrap();
                csc_a.1[start..end]
                    .iter()
                    .zip(&csc_a.2[start..end])
                    .filter(|&(&r, _)| r == row)
                    .map(|(_, &v)| v)
                    .sum()
            };

            // ThermalGeneration(10): +1.0 on thermal 10's column.
            let thermal_col = grid.flat(layout_a.equipment.thermal.start, t_pos, blk);
            assert_eq!(
                coeff_at(thermal_col),
                1.0,
                "blk {blk}: generic row must carry +1.0 on thermal 10's column \
                 (resolver path through thermal_pos)"
            );
            // LineExchange(100): +1.0 on the forward column, -1.0 on the reverse.
            assert_eq!(
                coeff_at(layout_a.line_fwd_col(LineSys::new(l_pos), blk)),
                1.0,
                "blk {blk}: generic row must carry +1.0 on line 100's forward column \
                 (resolver path through line_pos)"
            );
            assert_eq!(
                coeff_at(layout_a.line_rev_col(LineSys::new(l_pos), blk)),
                -1.0,
                "blk {blk}: generic row must carry -1.0 on line 100's reverse column"
            );
            // BusDeficit(2): +1.0 on each of bus 2's two deficit-segment columns.
            for seg in 0..2 {
                assert_eq!(
                    coeff_at(layout_a.deficit_col(b_pos, seg, blk)),
                    1.0,
                    "blk {blk}: generic row must carry +1.0 on bus 2 deficit segment {seg} \
                     (resolver path through bus_pos)"
                );
            }
        }
    }

    /// Declaration-order invariance on a FILLING system: build a two-filling-hydro
    /// system twice with the hydros declared in opposite orders, at a Filling stage,
    /// and assert BOTH the assembled CSC and the row bounds (including the per-stage
    /// `filling_target` `V_target[t]` RHS, which rides on the precomputed
    /// `ctx.filling_v_target`) are byte-identical. The two hydros carry DISTINCT
    /// resolved `min_storage_hm3` so a permutation that mislabelled which hydro owns
    /// which `σ_fill` floor would change the RHS and diverge here. This is the
    /// filling counterpart of `csc_byte_identical_under_permuted_declaration_order`.
    #[test]
    fn filling_csc_and_rows_byte_identical_under_permuted_declaration_order() {
        // Distinct dead volumes so a hydro mislabel changes the V_target RHS.
        const H1_MIN_STORAGE: f64 = 41.0;
        const H2_MIN_STORAGE: f64 = 58.0;

        // Build the fixture with the two filling hydros (ids 1, 2) declared in the
        // given order, set each hydro's resolved dead volume, and return the
        // assembled CSC plus the (row_lower, row_upper) bounds at a Filling stage.
        #[allow(clippy::type_complexity)]
        let build = |hydros: Vec<Hydro>| -> ((Vec<i32>, Vec<i32>, Vec<f64>), Vec<f64>, Vec<f64>) {
            let mut fixtures = PumpFixtures::new(hydros, Vec::new());
            let h1_idx = fixtures.hydro_pos[&EntityId(1)];
            let h2_idx = fixtures.hydro_pos[&EntityId(2)];
            fixtures.bounds.hydro_bounds_mut(h1_idx, 0).min_storage_hm3 = H1_MIN_STORAGE;
            fixtures.bounds.hydro_bounds_mut(h2_idx, 0).min_storage_hm3 = H2_MIN_STORAGE;
            let ctx = fixtures.make_ctx();
            // RET_FILLING_ID = 3 is a Filling stage for the start=2/entry=4 window.
            let stage = two_block_stage(usize::try_from(RET_FILLING_ID).unwrap(), [300.0, 444.0]);
            let state = state_layout_for(&ctx);
            let layout = StageLayout::new(&ctx, &state, &stage, 0);
            let (row_lower, row_upper) =
                super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
            let csc = {
                let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
                for col in &mut entries {
                    col.sort_unstable_by_key(|&(row, _)| row);
                }
                assemble_csc(&entries)
            };
            (csc, row_lower, row_upper)
        };

        // Both hydros are filling (start = 2, entry = 4); ret_hydro pins that window.
        let (csc_a, rl_a, ru_a) = build(vec![
            ret_hydro(1, Some(2), Some(RET_ENTRY_STAGE_ID), true),
            ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
        ]);
        let (csc_b, rl_b, ru_b) = build(vec![
            ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ret_hydro(1, Some(2), Some(RET_ENTRY_STAGE_ID), true),
        ]);

        assert_eq!(csc_a.0, csc_b.0, "col_starts must be byte-identical");
        assert_eq!(csc_a.1, csc_b.1, "row_indices must be byte-identical");
        assert_eq!(csc_a.2, csc_b.2, "values must be byte-identical");
        assert_eq!(
            rl_a, rl_b,
            "row_lower (incl. V_target RHS) must be byte-identical"
        );
        assert_eq!(ru_a, ru_b, "row_upper must be byte-identical");
    }

    /// Pin the two structural water-row coefficients `fill_state_and_water_entries`
    /// writes for a two-reservoir cascade: the cascade-upstream `−tau_h` and the
    /// AR-lag `−ζ·ψ`. Both are the weakest-backstopped coefficients in the water
    /// row — a sign flip on either silently mis-routes water and produces wrong
    /// bounds, yet outside this test they are exercised only by a slow parity
    /// D-case whose hash-mismatch failure mode does not localize to the water row.
    ///
    /// Cascade `H_up`(id 1) → `H_down`(id 2), both constant-productivity. On the
    /// downstream hydro's water-balance row, per block, the upstream turbine and
    /// spillage columns carry `−tau_h` (the upstream release arriving as inflow)
    /// while the downstream's own turbine carries `+tau_h` (its own outflow). The
    /// `+τ`/`−τ` sign *contrast* is asserted, not just one magnitude: a global
    /// flip negating both would otherwise pass. `tau_h` is computed from
    /// `M3S_TO_HM3` (never a literal) so the assertion cannot drift from the
    /// production constant.
    #[test]
    fn cascade_upstream_tau_and_ar_lag_land_on_downstream_water_row() {
        use cobre_core::scenario::InflowModel;

        // Assemble the production water-row fill into a CSC. The generic-constraint
        // fill is omitted (no generic constraints here); the water row is reached
        // through `build_stage_matrix_entries` alone.
        // H_up id 1 sorts to position 0, H_down id 2 to position 1.
        let up = 1;
        let down = 2;
        let cascade_fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );

        // Run the production water-row fill into a CSC. `ctx`/`state`/`layout` are
        // held by the test (borrowing `cascade_fixtures`, which outlives them), so
        // the layout offsets read below stay valid.
        let ctx = cascade_fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            // Mirror the production per-column row-sort (see
            // build_single_stage_template) before assembling the CSC.
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };

        // Sum the CSC values landing on `(col, row)` — mirrors the permutation
        // test's probe; multiple per-block pushes to one cell would otherwise hide.
        let coeff_at = |col: usize, row: i32| -> f64 {
            let start = usize::try_from(csc.0[col]).unwrap();
            let end = usize::try_from(csc.0[col + 1]).unwrap();
            csc.1[start..end]
                .iter()
                .zip(&csc.2[start..end])
                .filter(|&(&r, _)| r == row)
                .map(|(_, &v)| v)
                .sum()
        };

        let up_idx = 0; // H_up id 1.
        let down_idx = 1; // H_down id 2.
        let down_row = i32::try_from(layout.rows.water_balance.start + down_idx).unwrap();
        for blk in 0..layout.n_blks {
            // tau_h is the identical expression the production fill uses; the two
            // blocks carry distinct durations (300 vs 444), so a per-block divisor
            // confusion is observable.
            let tau_h = two_block_stage(0, [300.0, 444.0]).blocks[blk].duration_hours * M3S_TO_HM3;

            assert_eq!(
                coeff_at(layout.turbine_col(HydroSys::new(up_idx), blk), down_row),
                -tau_h,
                "blk {blk}: upstream turbine column must carry -tau_h on the \
                 downstream water row (cascade-upstream inflow)"
            );
            assert_eq!(
                coeff_at(layout.spillage_col(HydroSys::new(up_idx), blk), down_row),
                -tau_h,
                "blk {blk}: upstream spillage column must carry -tau_h on the \
                 downstream water row (cascade-upstream inflow)"
            );
            assert_eq!(
                coeff_at(layout.turbine_col(HydroSys::new(down_idx), blk), down_row),
                tau_h,
                "blk {blk}: downstream's OWN turbine column must carry +tau_h on \
                 its water row (self-outflow) — the +τ/−τ sign contrast"
            );
        }

        // AR-lag −ζ·ψ: self-contained block. An AR(1) PrecomputedPar carrying a
        // nonzero psi for the downstream hydro makes the AR-lag water term fire
        // (the default fixture has psi == 0, so the term is otherwise dormant).
        // psi[0] for the downstream hydro is constructed to equal phi exactly:
        // the classical conversion psi = phi * s_m / s_lag collapses to phi when
        // both the study stage and its pre-study lag stage carry the same std.
        let phi = 0.6_f64;
        let inflow_models = vec![
            // Upstream (id 1): white noise — psi stays 0 at every lag.
            InflowModel {
                hydro_id: EntityId(up),
                stage_id: 0,
                mean_m3s: 50.0,
                std_m3s: 1.0,
                ar_coefficients: vec![],
                residual_std_ratio: 1.0,
                annual: None,
            },
            // Downstream (id 2) study stage: AR(1) with coefficient phi; std == 1.0
            // so the lag-stage ratio is exactly 1.0.
            InflowModel {
                hydro_id: EntityId(down),
                stage_id: 0,
                mean_m3s: 80.0,
                std_m3s: 1.0,
                ar_coefficients: vec![phi],
                residual_std_ratio: 1.0,
                annual: None,
            },
            // Downstream pre-study lag stage (stage_id -1) with the same std, so
            // the Tier-1 exact-match lag lookup yields s_lag == s_m and the
            // conversion collapses to psi[0] == phi bit-exactly.
            InflowModel {
                hydro_id: EntityId(down),
                stage_id: -1,
                mean_m3s: 80.0,
                std_m3s: 1.0,
                ar_coefficients: vec![phi],
                residual_std_ratio: 1.0,
                annual: None,
            },
        ];
        let par_stage = two_block_stage(0, [300.0, 444.0]);
        let par_lp = PrecomputedPar::build(
            &inflow_models,
            std::slice::from_ref(&par_stage),
            &[EntityId(up), EntityId(down)],
            None,
        )
        .expect("AR(1) PrecomputedPar build must succeed");
        let psi_val = par_lp.psi_slice(0, down_idx)[0];
        assert_eq!(psi_val, phi, "downstream psi[0] must equal phi exactly");

        let ar_fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        )
        .with_par_lp(par_lp);
        let ar_ctx = ar_fixtures.make_ctx();
        let ar_stage = two_block_stage(0, [300.0, 444.0]);
        let ar_state = state_layout_for(&ar_ctx);
        let ar_layout = StageLayout::new(&ar_ctx, &ar_state, &ar_stage, 0);
        let ar_csc = {
            let mut entries = build_stage_matrix_entries(&ar_ctx, &ar_stage, 0, &ar_layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let ar_coeff_at = |col: usize, row: i32| -> f64 {
            let start = usize::try_from(ar_csc.0[col]).unwrap();
            let end = usize::try_from(ar_csc.0[col + 1]).unwrap();
            ar_csc.1[start..end]
                .iter()
                .zip(&ar_csc.2[start..end])
                .filter(|&(&r, _)| r == row)
                .map(|(_, &v)| v)
                .sum()
        };
        // Lag column for (lag 0, downstream hydro): col_inflow_lags_start + 0*n_h + h.
        let lag_col = ar_layout.col_inflow_lags_start() + down_idx;
        let ar_row = i32::try_from(ar_layout.rows.water_balance.start + down_idx).unwrap();
        assert_eq!(
            ar_coeff_at(lag_col, ar_row),
            -(ar_layout.zeta * psi_val),
            "downstream inflow-lag column must carry -(zeta * psi) on its water row"
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Water travel-time: parallel-mode arrival split + bucket definition rows
    // ─────────────────────────────────────────────────────────────────────────

    /// Sort each column's entries by row (mirroring `build_single_stage_template`)
    /// and assemble the CSC.
    fn build_sorted_csc(
        ctx: &TemplateBuildCtx<'_>,
        stage: &Stage,
        stage_idx: usize,
        layout: &StageLayout,
    ) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
        let mut entries = build_stage_matrix_entries(ctx, stage, stage_idx, layout);
        for col in &mut entries {
            col.sort_unstable_by_key(|&(row, _)| row);
        }
        assemble_csc(&entries)
    }

    /// Sum the CSC values landing on `(col, row)`; multiple per-block pushes to
    /// one cell would otherwise hide behind a single positional read.
    fn coeff_at(csc: &(Vec<i32>, Vec<i32>, Vec<f64>), col: usize, row: i32) -> f64 {
        let start = usize::try_from(csc.0[col]).unwrap();
        let end = usize::try_from(csc.0[col + 1]).unwrap();
        csc.1[start..end]
            .iter()
            .zip(&csc.2[start..end])
            .filter(|&(&r, _)| r == row)
            .map(|(_, &v)| v)
            .sum()
    }

    /// One declared arc `k = [1/2, 1/2]` (depth 1, the plant's only bucket):
    /// the downstream balance row carries `-1/2 * tau_h` on the upstream's own
    /// turbine/spillage columns (NOT `-tau_h`) plus `-1.0` on the plant's
    /// `b_1^in` column, and the single bucket-definition row deposits
    /// `-1/2 * tau_h` into the same upstream columns with `+1.0` on `b_1^out` —
    /// `b_{L+1}^in` does not exist since `d == L_j`.
    #[test]
    fn declared_arc_arrival_split_and_single_definition_row() {
        let up = 1;
        let down = 2;
        let fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        let mut arc_stage_weights = HashMap::new();
        arc_stage_weights.insert(up_idx, vec![vec![0.5, 0.5]]);
        let ctx = TemplateBuildCtx {
            arc_stage_weights,
            per_stage_mask: vec![vec![1]],
            ..fixtures.make_ctx()
        };

        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = StateLayout::new(
            ctx.n_hydros,
            ctx.max_par_order,
            1,
            vec![(down_idx, 1)],
            ctx.n_anticipated,
            ctx.k_max,
            ctx.anticipated_lead_stages.clone(),
            &vec![0; ctx.n_hydros],
        );
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);

        let down_row = i32::try_from(layout.rows.water_balance.start + down_idx).unwrap();
        let def_row = i32::try_from(layout.rows.transit_bucket_definition.start).unwrap();
        let col_first_slot_in = state.transit_buckets_in.start;
        let col_first_slot_out = state.transit_buckets_out.start;

        for blk in 0..layout.n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.turbine_col(HydroSys::new(up_idx), blk),
                    down_row
                ),
                -0.5 * tau_h,
                "blk {blk}: balance row must carry -k_0*tau_h, not -tau_h"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.spillage_col(HydroSys::new(up_idx), blk),
                    down_row
                ),
                -0.5 * tau_h,
                "blk {blk}: balance row spillage must carry -k_0*tau_h"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.turbine_col(HydroSys::new(up_idx), blk),
                    def_row
                ),
                -0.5 * tau_h,
                "blk {blk}: definition row must carry -k_1*tau_h on the SAME release column"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.spillage_col(HydroSys::new(up_idx), blk),
                    def_row
                ),
                -0.5 * tau_h,
                "blk {blk}: definition row spillage must carry -k_1*tau_h"
            );
        }
        assert_eq!(
            coeff_at(&csc, col_first_slot_in, down_row),
            -1.0,
            "the maturing-now bucket b_1^in must carry -1.0 on the balance row"
        );
        assert_eq!(
            coeff_at(&csc, col_first_slot_out, def_row),
            1.0,
            "the definition row must carry +1.0 on b_1^out"
        );
        // b_{L+1}^in does not exist: the plant's only bucket is depth 1, so
        // transit_buckets_in has exactly one column (no b_2^in to reference).
        assert_eq!(state.transit_buckets_in.len(), 1);
    }

    /// Row 13 (Filling arm): a `Filling`-phase upstream (turbine/diversion
    /// frozen, spillage FREE — the D40 relief valve) still deposits its
    /// spillage share into the downstream balance row and the bucket
    /// definition row — `fill_arc_release_block_entries` never special-cases
    /// the RELEASING hydro's own commissioning phase, so the `(u+s)` deposit
    /// rides unchanged. Cross-checked against `columns.rs`: the coefficient
    /// carries real flow because the spillage column is actually free at this
    /// stage, not a coefficient on a column pinned to zero.
    #[test]
    fn filling_upstream_spillage_still_deposits_into_transit_bucket() {
        use cobre_core::entities::hydro::FillingConfig;

        let up = 1;
        let down = 2;
        let mut up_hydro = fixture_hydro_ds(up, Some(down));
        up_hydro.filling = Some(FillingConfig {
            start_stage_id: 0,
            filling_min_rate_m3s: 0.0,
        });
        up_hydro.entry_stage_id = Some(5);
        let fixtures = PumpFixtures::new_full(
            vec![up_hydro, fixture_hydro_ds(down, None)],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        )
        .with_resolved_penalties();
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        let mut arc_stage_weights = HashMap::new();
        arc_stage_weights.insert(up_idx, vec![vec![0.5, 0.5]]);
        let ctx = TemplateBuildCtx {
            arc_stage_weights,
            per_stage_mask: vec![vec![1]],
            ..fixtures.make_ctx()
        };

        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = StateLayout::new(
            ctx.n_hydros,
            ctx.max_par_order,
            1,
            vec![(down_idx, 1)],
            ctx.n_anticipated,
            ctx.k_max,
            ctx.anticipated_lead_stages.clone(),
            &vec![0; ctx.n_hydros],
        );
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);

        let down_row = i32::try_from(layout.rows.water_balance.start + down_idx).unwrap();
        let def_row = i32::try_from(layout.rows.transit_bucket_definition.start).unwrap();

        for blk in 0..layout.n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.spillage_col(HydroSys::new(up_idx), blk),
                    down_row
                ),
                -0.5 * tau_h,
                "blk {blk}: a Filling upstream's spillage must still carry -k_0*tau_h on the \
                 balance row"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.spillage_col(HydroSys::new(up_idx), blk),
                    def_row
                ),
                -0.5 * tau_h,
                "blk {blk}: a Filling upstream's spillage must still deposit -k_1*tau_h into \
                 the bucket"
            );
        }

        let (_col_lower, col_upper, _objective) =
            super::super::columns::fill_stage_columns(&ctx, &stage, 0, &layout);
        for blk in 0..layout.n_blks {
            assert_eq!(
                col_upper[layout.spillage_col(HydroSys::new(up_idx), blk)],
                f64::INFINITY,
                "blk {blk}: a Filling upstream's spillage column must stay free (D40), not frozen"
            );
        }
    }

    /// Row 13 (exit arm): an upstream past its own `exit_stage_id` still
    /// emits the SAME arc-release deposit coefficients as an active upstream
    /// — `fill_arc_release_block_entries` never special-cases the RELEASING
    /// hydro's own commissioning phase. Combined with the commissioning
    /// freeze pinning its turbine/spillage columns to `[0, 0]` post-exit
    /// (verified below via `columns.rs`), the realized deposit is zero with
    /// no special-cased code path, so the bucket drains through the pure
    /// ring shift (the state-assembly copy-gap) with no replenishment.
    #[test]
    fn exited_upstream_arc_deposit_coefficients_unchanged_no_special_case() {
        let up = 1;
        let down = 2;
        let mut up_hydro = fixture_hydro_ds(up, Some(down));
        up_hydro.exit_stage_id = Some(1);
        let fixtures = PumpFixtures::new_full(
            vec![up_hydro, fixture_hydro_ds(down, None)],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        )
        .with_resolved_penalties();
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        let mut arc_stage_weights = HashMap::new();
        arc_stage_weights.insert(up_idx, vec![vec![0.5, 0.5]]);
        let ctx = TemplateBuildCtx {
            arc_stage_weights,
            per_stage_mask: vec![vec![1]],
            ..fixtures.make_ctx()
        };
        let state = StateLayout::new(
            ctx.n_hydros,
            ctx.max_par_order,
            1,
            vec![(down_idx, 1)],
            ctx.n_anticipated,
            ctx.k_max,
            ctx.anticipated_lead_stages.clone(),
            &vec![0; ctx.n_hydros],
        );

        // `stage_idx` stays 0 for both builds (the single-stage fixture's
        // established decoupling from `stage.id`); only `stage.id` moves
        // across the upstream's `exit_stage_id` boundary.
        let mut stage_active = two_block_stage(0, [300.0, 444.0]);
        stage_active.id = 0;
        let layout_active = StageLayout::new(&ctx, &state, &stage_active, 0);
        let csc_active = build_sorted_csc(&ctx, &stage_active, 0, &layout_active);

        let mut stage_exited = two_block_stage(0, [300.0, 444.0]);
        stage_exited.id = 1;
        let layout_exited = StageLayout::new(&ctx, &state, &stage_exited, 0);
        let csc_exited = build_sorted_csc(&ctx, &stage_exited, 0, &layout_exited);

        let down_row_active =
            i32::try_from(layout_active.rows.water_balance.start + down_idx).unwrap();
        let def_row_active =
            i32::try_from(layout_active.rows.transit_bucket_definition.start).unwrap();
        let down_row_exited =
            i32::try_from(layout_exited.rows.water_balance.start + down_idx).unwrap();
        let def_row_exited =
            i32::try_from(layout_exited.rows.transit_bucket_definition.start).unwrap();

        for blk in 0..layout_active.n_blks {
            for (col_active, col_exited) in [
                (
                    layout_active.turbine_col(HydroSys::new(up_idx), blk),
                    layout_exited.turbine_col(HydroSys::new(up_idx), blk),
                ),
                (
                    layout_active.spillage_col(HydroSys::new(up_idx), blk),
                    layout_exited.spillage_col(HydroSys::new(up_idx), blk),
                ),
            ] {
                assert_eq!(
                    coeff_at(&csc_active, col_active, down_row_active),
                    coeff_at(&csc_exited, col_exited, down_row_exited),
                    "blk {blk}: the balance-row deposit coefficient must not depend on the \
                     upstream's own exit_stage_id"
                );
                assert_eq!(
                    coeff_at(&csc_active, col_active, def_row_active),
                    coeff_at(&csc_exited, col_exited, def_row_exited),
                    "blk {blk}: the bucket-definition deposit coefficient must not depend on \
                     the upstream's own exit_stage_id"
                );
            }
        }

        // The freeze that actually zeroes the realized deposit lives in
        // columns.rs, not here: post-exit, both columns are pinned [0, 0].
        let (_lo_active, col_upper_active, _obj_active) =
            super::super::columns::fill_stage_columns(&ctx, &stage_active, 0, &layout_active);
        let (_lo_exited, col_upper_exited, _obj_exited) =
            super::super::columns::fill_stage_columns(&ctx, &stage_exited, 0, &layout_exited);
        for blk in 0..layout_active.n_blks {
            assert!(
                col_upper_active[layout_active.turbine_col(HydroSys::new(up_idx), blk)] > 0.0,
                "blk {blk}: the active upstream's turbine column must be free before exit"
            );
            assert_eq!(
                col_upper_exited[layout_exited.turbine_col(HydroSys::new(up_idx), blk)],
                0.0,
                "blk {blk}: the exited upstream's turbine column must be pinned to 0"
            );
            assert_eq!(
                col_upper_exited[layout_exited.spillage_col(HydroSys::new(up_idx), blk)],
                0.0,
                "blk {blk}: the exited upstream's spillage column must be pinned to 0 \
                 (post-exit reverts to PreFilling, which freezes spillage too)"
            );
        }
    }

    /// Confluence: two upstreams feeding one downstream plant sum their
    /// travel-time deposits into the SAME per-plant bucket definition row,
    /// not one row per arc.
    #[test]
    fn confluence_two_upstreams_sum_deposits_into_single_definition_row() {
        let up_a = 1;
        let up_b = 2;
        let down = 3;
        let fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up_a, Some(down)),
                fixture_hydro_ds(up_b, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_a_idx = fixtures.hydro_pos[&EntityId(up_a)];
        let up_b_idx = fixtures.hydro_pos[&EntityId(up_b)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        let mut arc_stage_weights = HashMap::new();
        arc_stage_weights.insert(up_a_idx, vec![vec![0.5, 0.5]]);
        arc_stage_weights.insert(up_b_idx, vec![vec![0.25, 0.75]]);
        let ctx = TemplateBuildCtx {
            arc_stage_weights,
            per_stage_mask: vec![vec![1]],
            ..fixtures.make_ctx()
        };

        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = StateLayout::new(
            ctx.n_hydros,
            ctx.max_par_order,
            1,
            vec![(down_idx, 1)],
            ctx.n_anticipated,
            ctx.k_max,
            ctx.anticipated_lead_stages.clone(),
            &vec![0; ctx.n_hydros],
        );
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);

        let def_row = i32::try_from(layout.rows.transit_bucket_definition.start).unwrap();
        for blk in 0..layout.n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.turbine_col(HydroSys::new(up_a_idx), blk),
                    def_row
                ),
                -0.5 * tau_h,
                "blk {blk}: upstream A's deposit must land on the shared definition row"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.spillage_col(HydroSys::new(up_a_idx), blk),
                    def_row
                ),
                -0.5 * tau_h
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.turbine_col(HydroSys::new(up_b_idx), blk),
                    def_row
                ),
                -0.75 * tau_h,
                "blk {blk}: upstream B's deposit must land on the SAME shared definition row"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.spillage_col(HydroSys::new(up_b_idx), blk),
                    def_row
                ),
                -0.75 * tau_h
            );
        }
        // A single aggregated bucket for the downstream plant, not one per arc.
        assert_eq!(state.n_buckets, 1);
        assert_eq!(state.transit_buckets_out.len(), 1);
    }

    /// Equivalence pin: `fill_transit_bucket_definition_entries`'s per-plant
    /// `DeliveryRing::emit_shift_rows` routing reproduces the open-coded reference
    /// formula (`+1` on `b_d^out`, `-1` on `b_{d+1}^in` only within the SAME plant's
    /// own contiguous group) on a fixed three-plant fixture: one plant with 3 lags,
    /// one with 1 lag, and one with no declared arc at all (absent from `column_order`).
    #[test]
    fn fill_transit_bucket_definition_entries_matches_pre_migration_formula_across_ragged_plants() {
        let h_down3 = 10;
        let h_down1 = 20;
        let h_none = 30;
        let fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(h_down3, None),
                fixture_hydro_ds(h_down1, None),
                fixture_hydro_ds(h_none, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let down3_idx = fixtures.hydro_pos[&EntityId(h_down3)];
        let down1_idx = fixtures.hydro_pos[&EntityId(h_down1)];

        let ctx = TemplateBuildCtx {
            per_stage_mask: vec![vec![3, 1]],
            ..fixtures.make_ctx()
        };
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = StateLayout::new(
            ctx.n_hydros,
            ctx.max_par_order,
            4,
            vec![
                (down3_idx, 1),
                (down3_idx, 2),
                (down3_idx, 3),
                (down1_idx, 1),
            ],
            ctx.n_anticipated,
            ctx.k_max,
            ctx.anticipated_lead_stages.clone(),
            &vec![0; ctx.n_hydros],
        );
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        fill_transit_bucket_definition_entries(&layout, &mut col_entries);

        let row_start = layout.rows.transit_bucket_definition.start;
        let out = state.transit_buckets_out.start;
        let inn = state.transit_buckets_in.start;

        // down3 slot 0 (lag 1): a deeper own-plant slot (lag 2) exists.
        assert_eq!(col_entries[out], vec![(row_start, 1.0)]);
        assert_eq!(col_entries[inn + 1], vec![(row_start, -1.0)]);
        // down3 slot 1 (lag 2): a deeper own-plant slot (lag 3) exists.
        assert_eq!(col_entries[out + 1], vec![(row_start + 1, 1.0)]);
        assert_eq!(col_entries[inn + 2], vec![(row_start + 1, -1.0)]);
        // down3 slot 2 (lag 3, its own last lag): the next global slot belongs
        // to down1, so no shift term crosses the plant boundary.
        assert_eq!(col_entries[out + 2], vec![(row_start + 2, 1.0)]);
        assert!(col_entries[inn + 3].is_empty());
        // down1 slot 3 (lag 1, its only lag, also the last global slot): no
        // deeper slot exists at all.
        assert_eq!(col_entries[out + 3], vec![(row_start + 3, 1.0)]);
    }

    /// `B == 0` (no declared arc, `state.n_buckets == 0`): the emitted water
    /// entries are byte-identical to today's shape — no bucket-definition rows
    /// exist at all (`load_balance` collapses back onto `water_balance.end`)
    /// and the upstream release carries exactly `-tau_h` (the `n_buckets` == 0
    /// byte-identity anchor).
    #[test]
    fn b_zero_water_entries_are_byte_identical_to_undeclared_arc() {
        let up = 1;
        let down = 2;
        let fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        assert_eq!(
            state.n_buckets, 0,
            "fixture must declare no travel-time arc"
        );
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        // No bucket-row gap: load_balance starts exactly where water_balance
        // ends, and the bucket-definition row cursor collapses onto it.
        assert_eq!(
            layout.rows.transit_bucket_definition.start, layout.rows.load_balance.start,
            "B==0 must leave no bucket-definition rows between water_balance and load_balance"
        );
        assert_eq!(
            layout.rows.load_balance.start,
            layout.rows.water_balance.start + layout.n_h,
            "B==0 must reproduce today's row_water_balance_start + n_hydros offset"
        );

        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);
        let down_row = i32::try_from(layout.rows.water_balance.start + down_idx).unwrap();
        for blk in 0..layout.n_blks {
            let tau_h = stage.blocks[blk].duration_hours * M3S_TO_HM3;
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.turbine_col(HydroSys::new(up_idx), blk),
                    down_row
                ),
                -tau_h,
                "blk {blk}: undeclared arc must carry exactly -tau_h (today's shape)"
            );
            assert_eq!(
                coeff_at(
                    &csc,
                    layout.spillage_col(HydroSys::new(up_idx), blk),
                    down_row
                ),
                -tau_h,
                "blk {blk}: undeclared arc must carry exactly -tau_h (today's shape)"
            );
        }
    }

    /// The conservation `debug_assert` fires when a declared arc's `k` does not
    /// sum to 1.0 — the guard is real, not dead code.
    #[test]
    #[should_panic(expected = "stage-clock weights must sum to 1.0")]
    fn declared_arc_non_conserving_k_panics_in_debug() {
        let up = 1;
        let down = 2;
        let fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        let mut arc_stage_weights = HashMap::new();
        arc_stage_weights.insert(up_idx, vec![vec![0.5, 0.3]]); // sums to 0.8: violates conservation.
        let ctx = TemplateBuildCtx {
            arc_stage_weights,
            per_stage_mask: vec![vec![1]],
            ..fixtures.make_ctx()
        };

        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = StateLayout::new(
            ctx.n_hydros,
            ctx.max_par_order,
            1,
            vec![(down_idx, 1)],
            ctx.n_anticipated,
            ctx.k_max,
            ctx.anticipated_lead_stages.clone(),
            &vec![0; ctx.n_hydros],
        );
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let _ = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Water travel-time: chronological block-resolved deposits/routing/delivery
    // ─────────────────────────────────────────────────────────────────────────

    /// A chronological `Stage` with the given per-block durations, otherwise
    /// mirroring [`two_block_stage`]'s fixture defaults.
    fn chronological_stage(index: usize, block_hours: &[f64]) -> Stage {
        let mut stage = two_block_stage(
            index,
            [
                block_hours[0],
                block_hours.get(1).copied().unwrap_or(block_hours[0]),
            ],
        );
        stage.blocks = block_hours
            .iter()
            .enumerate()
            .map(|(i, &h)| cobre_core::Block {
                index: i,
                name: format!("B{i}"),
                duration_hours: h,
            })
            .collect();
        stage.block_mode = cobre_core::BlockMode::Chronological;
        stage
    }

    /// A `t_v = 250h` travel time against a 720h stage of 3×240h
    /// chronological blocks. Pins
    /// `κ_{B0→B1}=230/240, κ_{B0→B2}=10/240, κ_{B1→B2}=230/240,
    /// χ=(0, 10/240, 1)`-shaped on the emitted matrix entries.
    #[test]
    fn example_iii_kappa_and_chi_match_worked_numbers() {
        let up = 1;
        let down = 2;
        let fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        let t_v = 250.0_f64;
        let block_hours = [240.0, 240.0, 240.0];
        let resolution = resolve_spread(t_v, 0, &[720.0, 720.0], Some(&block_hours));

        assert!((resolution.within_stage_routing[0][1] - 230.0 / 240.0).abs() < 1e-9);
        assert!((resolution.within_stage_routing[0][2] - 10.0 / 240.0).abs() < 1e-9);
        assert!((resolution.within_stage_routing[1][1] - 230.0 / 240.0).abs() < 1e-9);
        assert!(resolution.block_deposits[0][1].abs() < 1e-9);
        assert!((resolution.block_deposits[1][1] - 10.0 / 240.0).abs() < 1e-9);
        assert!((resolution.block_deposits[2][1] - 1.0).abs() < 1e-9);

        let mut arc_spread_chrono = HashMap::new();
        arc_spread_chrono.insert(up_idx, vec![Some(resolution)]);
        let ctx = TemplateBuildCtx {
            arc_spread_chrono,
            per_stage_mask: vec![vec![1]],
            ..fixtures.make_ctx()
        };

        let stage = chronological_stage(0, &block_hours);
        let state = StateLayout::new(
            ctx.n_hydros,
            ctx.max_par_order,
            1,
            vec![(down_idx, 1)],
            ctx.n_anticipated,
            ctx.k_max,
            ctx.anticipated_lead_stages.clone(),
            &vec![0; ctx.n_hydros],
        );
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);

        let def_row = i32::try_from(layout.rows.transit_bucket_definition.start).unwrap();
        let row_water = layout.rows.water_balance.start;
        let row_b1 = i32::try_from(row_water + down_idx * 3 + 1).unwrap();
        let row_b2 = i32::try_from(row_water + down_idx * 3 + 2).unwrap();
        let tau = |b: usize| stage.blocks[b].duration_hours * M3S_TO_HM3;

        // Block B0: routes 230/240 to B1, 10/240 to B2; no crossing deposit.
        assert_eq!(
            coeff_at(&csc, layout.turbine_col(HydroSys::new(up_idx), 0), row_b1),
            -(230.0 / 240.0) * tau(0)
        );
        assert_eq!(
            coeff_at(&csc, layout.spillage_col(HydroSys::new(up_idx), 0), row_b1),
            -(230.0 / 240.0) * tau(0)
        );
        assert_eq!(
            coeff_at(&csc, layout.turbine_col(HydroSys::new(up_idx), 0), row_b2),
            -(10.0 / 240.0) * tau(0)
        );
        assert_eq!(
            coeff_at(&csc, layout.turbine_col(HydroSys::new(up_idx), 0), def_row),
            0.0
        );

        // Block B1: routes 230/240 to B2; deposits 10/240 into the bucket.
        assert_eq!(
            coeff_at(&csc, layout.turbine_col(HydroSys::new(up_idx), 1), row_b2),
            -(230.0 / 240.0) * tau(1)
        );
        assert_eq!(
            coeff_at(&csc, layout.turbine_col(HydroSys::new(up_idx), 1), def_row),
            -(10.0 / 240.0) * tau(1)
        );

        // Block B2: nothing in-stage; deposits its full release into the bucket.
        assert_eq!(
            coeff_at(&csc, layout.turbine_col(HydroSys::new(up_idx), 2), def_row),
            -tau(2)
        );
        assert_eq!(
            coeff_at(&csc, layout.spillage_col(HydroSys::new(up_idx), 2), def_row),
            -tau(2)
        );
    }

    /// `K = 1` (a single block spanning the whole stage) with travel time ON is
    /// byte-identical to the parallel fill (the K=1 parity anchor):
    /// `χ_{0,d} = k_d` collapses κ to self-routing and χ to the parallel deposit.
    #[test]
    fn k1_chronological_with_travel_time_is_byte_identical_to_parallel() {
        let up = 1;
        let down = 2;
        let t_v = 250.0_f64;
        let stage_durations = [720.0, 720.0];

        let make_fixtures = || {
            PumpFixtures::new_full(
                vec![
                    fixture_hydro_ds(up, Some(down)),
                    fixture_hydro_ds(down, None),
                ],
                Vec::new(),
                vec![fixture_bus(1)],
                Vec::new(),
                Vec::new(),
            )
        };
        let up_idx = make_fixtures().hydro_pos[&EntityId(up)];
        let down_idx = make_fixtures().hydro_pos[&EntityId(down)];

        let resolution = resolve_spread(t_v, 0, &stage_durations, Some(&[720.0]));
        let stage_weights = resolution.stage_weights.clone();

        // Parallel build. `chronological_stage` (not `two_block_stage`, which
        // always carries 2 blocks) gives a single 720h block so `n_blks == 1`
        // on both sides of the comparison; the mode is then forced back to
        // `Parallel` for this side.
        let par_fixtures = make_fixtures();
        let mut arc_stage_weights = HashMap::new();
        arc_stage_weights.insert(up_idx, vec![stage_weights]);
        let par_ctx = TemplateBuildCtx {
            arc_stage_weights,
            per_stage_mask: vec![vec![1]],
            ..par_fixtures.make_ctx()
        };
        let mut par_stage = chronological_stage(0, &[720.0]);
        par_stage.block_mode = cobre_core::BlockMode::Parallel;
        let par_state = StateLayout::new(
            par_ctx.n_hydros,
            par_ctx.max_par_order,
            1,
            vec![(down_idx, 1)],
            par_ctx.n_anticipated,
            par_ctx.k_max,
            par_ctx.anticipated_lead_stages.clone(),
            &vec![0; par_ctx.n_hydros],
        );
        let par_layout = StageLayout::new(&par_ctx, &par_state, &par_stage, 0);
        let par_csc = build_sorted_csc(&par_ctx, &par_stage, 0, &par_layout);

        // Chronological build (K=1), same arc data via arc_spread_chrono.
        let chr_fixtures = make_fixtures();
        let mut arc_spread_chrono = HashMap::new();
        arc_spread_chrono.insert(up_idx, vec![Some(resolution)]);
        let chr_ctx = TemplateBuildCtx {
            arc_spread_chrono,
            per_stage_mask: vec![vec![1]],
            ..chr_fixtures.make_ctx()
        };
        let chr_stage = chronological_stage(0, &[720.0]);
        let chr_state = StateLayout::new(
            chr_ctx.n_hydros,
            chr_ctx.max_par_order,
            1,
            vec![(down_idx, 1)],
            chr_ctx.n_anticipated,
            chr_ctx.k_max,
            chr_ctx.anticipated_lead_stages.clone(),
            &vec![0; chr_ctx.n_hydros],
        );
        let chr_layout = StageLayout::new(&chr_ctx, &chr_state, &chr_stage, 0);
        let chr_csc = build_sorted_csc(&chr_ctx, &chr_stage, 0, &chr_layout);

        assert_eq!(
            par_layout.num_cols, chr_layout.num_cols,
            "K=1 must carry the same column count in both modes"
        );
        assert_eq!(
            par_layout.rows.num_rows, chr_layout.rows.num_rows,
            "K=1 must carry the same row count in both modes"
        );
        assert_eq!(
            par_csc, chr_csc,
            "K=1 chronological with travel time ON must be byte-identical to parallel"
        );
    }

    /// The `Σ_d k_d == 1.0` stage-clock conservation (row 8) fires at the
    /// CHRONOLOGICAL deposit site too, independently of the per-column and
    /// aggregation identities below it — mirrors the parallel-site guard
    /// (`declared_arc_non_conserving_k_panics_in_debug`), closing the gap
    /// where only the chrono-specific identities were fill-time-asserted.
    #[test]
    #[should_panic(expected = "stage-clock weights must sum to 1.0")]
    fn row_8_chrono_stage_clock_sum_panics_on_disagreement() {
        let up = 1;
        let down = 2;
        let fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        // stage_weights = [0.3, 0.3] sums to 0.6 — violates row 8's
        // stage-clock conservation, even though within_stage_routing's/
        // block_deposits's own per-column identity (0.5 + 0.5 == 1.0) holds.
        let bad_resolution = SpreadResolution {
            stage_reach: 1,
            stage_weights: vec![0.3, 0.3],
            block_deposits: vec![vec![0.5, 0.5]],
            within_stage_routing: vec![vec![0.5]],
            arrival_density: vec![vec![1.0]],
        };
        let mut arc_spread_chrono = HashMap::new();
        arc_spread_chrono.insert(up_idx, vec![Some(bad_resolution)]);
        let ctx = TemplateBuildCtx {
            arc_spread_chrono,
            per_stage_mask: vec![vec![1]],
            ..fixtures.make_ctx()
        };

        let stage = chronological_stage(0, &[720.0]);
        let state = StateLayout::new(
            ctx.n_hydros,
            ctx.max_par_order,
            1,
            vec![(down_idx, 1)],
            ctx.n_anticipated,
            ctx.k_max,
            ctx.anticipated_lead_stages.clone(),
            &vec![0; ctx.n_hydros],
        );
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let _ = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
    }

    /// The `Σ_b w_b·χ_{b,d} == k_d` shared-density consistency (row 9) fires
    /// when a hand-built resolution's block deposits disagree with its own
    /// stage-level `stage_weights` — the guard is real, not dead code.
    #[test]
    #[should_panic(expected = "block deposits must aggregate to k_d")]
    fn row_9_shared_density_consistency_panics_on_disagreement() {
        let up = 1;
        let down = 2;
        let fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        // k_1 = 0.5 but the single block's block_deposits value is 0.0 at
        // lag 1 — violates the shared-density aggregation identity.
        let bad_resolution = SpreadResolution {
            stage_reach: 1,
            stage_weights: vec![0.5, 0.5],
            block_deposits: vec![vec![1.0, 0.0]],
            within_stage_routing: vec![vec![1.0]],
            arrival_density: vec![vec![1.0]],
        };
        let mut arc_spread_chrono = HashMap::new();
        arc_spread_chrono.insert(up_idx, vec![Some(bad_resolution)]);
        let ctx = TemplateBuildCtx {
            arc_spread_chrono,
            per_stage_mask: vec![vec![1]],
            ..fixtures.make_ctx()
        };

        let stage = chronological_stage(0, &[720.0]);
        let state = StateLayout::new(
            ctx.n_hydros,
            ctx.max_par_order,
            1,
            vec![(down_idx, 1)],
            ctx.n_anticipated,
            ctx.k_max,
            ctx.anticipated_lead_stages.clone(),
            &vec![0; ctx.n_hydros],
        );
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let _ = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
    }

    /// `resolve_chrono_arrival_density` returns the precomputed arrival-frame
    /// `arc_arrival_density` table entry verbatim — a lookup, not a
    /// re-derivation from the sender's own lag-1 row.
    #[test]
    fn resolve_chrono_arrival_density_looks_up_arrival_frame_table() {
        let up = 1;
        let down = 2;
        let fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];

        let table_density = vec![0.3, 0.7];
        let mut arc_arrival_density = HashMap::new();
        arc_arrival_density.insert(up_idx, vec![None, Some(table_density.clone())]);
        let ctx = TemplateBuildCtx {
            arc_arrival_density,
            ..fixtures.make_ctx()
        };

        let stage = chronological_stage(1, &[300.0, 420.0]);
        let resolved = resolve_chrono_arrival_density(&ctx, &stage, 1, EntityId(down), 2);

        assert_eq!(
            resolved, table_density,
            "must return the precomputed arrival-frame table entry, not a re-derived density"
        );
    }

    /// The genuine no-arc/first-stage default: the study's first stage has no
    /// source stage to blend from, so `arc_arrival_density` carries no entry
    /// (mirrors the real setup precompute's `None` at stage 0) and the
    /// fallback is the duration-weighted uniform density.
    #[test]
    fn resolve_chrono_arrival_density_falls_back_to_uniform_when_table_entry_absent() {
        let up = 1;
        let down = 2;
        let fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let ctx = fixtures.make_ctx();

        let stage = chronological_stage(0, &[300.0, 420.0]);
        let resolved = resolve_chrono_arrival_density(&ctx, &stage, 0, EntityId(down), 2);

        assert_eq!(
            resolved,
            vec![300.0 / 720.0, 420.0 / 720.0],
            "must fall back to the duration-weighted uniform density"
        );
    }

    /// A plain (no-travel-time) tributary into a bucketed confluence is EXCLUDED
    /// from the arrival-density resolution: `build_arc_arrival_density` inserts
    /// declared travel-time arcs only, so the plain tributary has no table entry
    /// and the resolver skips it, returning the sole travel-time arc's density —
    /// never a false heterogeneous-confluence panic (debug) nor a wrong uniform
    /// split (release). The plain tributary sorts first in `cascade.upstream`
    /// (id 0 < id 1), so the pre-fix code would seed `chosen` with the uniform
    /// fallback before the travel-time arc disagrees. The `cobre-io`
    /// `check_chronological_confluence_heterogeneous_travel_time` gate cannot
    /// catch this: it counts travel-time arcs only, so one arc plus one plain
    /// tributary is `< 2` and passes config validation.
    #[test]
    fn resolve_chrono_arrival_density_excludes_plain_tributary_from_confluence() {
        let plain = 0;
        let up = 1;
        let down = 2;
        let fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(plain, Some(down)),
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];

        // Only the travel-time arc is in the arrival-frame table, and its entry
        // is deliberately non-uniform so the pre-fix uniform fallback disagrees.
        let arrival_density = vec![0.25, 0.75];
        let mut arc_arrival_density = HashMap::new();
        arc_arrival_density.insert(up_idx, vec![None, Some(arrival_density.clone())]);
        let ctx = TemplateBuildCtx {
            arc_arrival_density,
            ..fixtures.make_ctx()
        };

        let stage = chronological_stage(1, &[300.0, 420.0]);
        let resolved = resolve_chrono_arrival_density(&ctx, &stage, 1, EntityId(down), 2);

        assert_eq!(
            resolved, arrival_density,
            "the plain tributary must be skipped; the travel-time arc's density wins"
        );
    }

    /// `fill_parallel_water_entries` never reads `arc_arrival_density`: the
    /// maturing bucket's parallel entry stays a single `-1.0` (the confluence
    /// sum lives in the state variable, not a density split) regardless of
    /// what the arrival-frame table holds for the arc at this stage.
    #[test]
    fn fill_parallel_water_entries_ignores_arc_arrival_density() {
        let up = 1;
        let down = 2;
        let fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        // A deliberately non-uniform entry: if the parallel fill read this
        // table at all, the bucket's maturing entry would carry something
        // other than -1.0.
        let mut arc_arrival_density = HashMap::new();
        arc_arrival_density.insert(up_idx, vec![Some(vec![0.9, 0.1])]);
        let ctx = TemplateBuildCtx {
            arc_arrival_density,
            per_stage_mask: vec![vec![1]],
            ..fixtures.make_ctx()
        };

        let stage = two_block_stage(0, [300.0, 420.0]);
        let state = StateLayout::new(
            ctx.n_hydros,
            ctx.max_par_order,
            1,
            vec![(down_idx, 1)],
            ctx.n_anticipated,
            ctx.k_max,
            ctx.anticipated_lead_stages.clone(),
            &vec![0; ctx.n_hydros],
        );
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let csc = build_sorted_csc(&ctx, &stage, 0, &layout);

        let row_water = i32::try_from(layout.rows.water_balance.start + down_idx).unwrap();
        let col_first_slot_in = state.transit_buckets_in.start;
        assert_eq!(
            coeff_at(&csc, col_first_slot_in, row_water),
            -1.0,
            "the parallel maturing-bucket entry must stay a single -1.0, \
             independent of arc_arrival_density"
        );
    }

    /// The `Σ_b arrival_density_b == 1` conservation `debug_assert` in
    /// [`fill_chronological_water_entries`] still fires when a hand-built
    /// `arc_arrival_density` table entry violates conservation — the guard is
    /// real, not dead code, after the arrival-frame lookup swap.
    #[test]
    #[should_panic(expected = "arrival_density must sum to 1.0")]
    fn fill_chronological_water_entries_arrival_density_conservation_panics_on_disagreement() {
        let up = 1;
        let down = 2;
        let fixtures = PumpFixtures::new_full(
            vec![
                fixture_hydro_ds(up, Some(down)),
                fixture_hydro_ds(down, None),
            ],
            Vec::new(),
            vec![fixture_bus(1)],
            Vec::new(),
            Vec::new(),
        );
        let up_idx = fixtures.hydro_pos[&EntityId(up)];
        let down_idx = fixtures.hydro_pos[&EntityId(down)];

        // Deliberately non-conserving: sums to 0.6, not 1.0.
        let mut arc_arrival_density = HashMap::new();
        arc_arrival_density.insert(up_idx, vec![Some(vec![0.3, 0.3])]);
        let ctx = TemplateBuildCtx {
            arc_arrival_density,
            per_stage_mask: vec![vec![1]],
            ..fixtures.make_ctx()
        };

        let stage = chronological_stage(0, &[300.0, 420.0]);
        let state = StateLayout::new(
            ctx.n_hydros,
            ctx.max_par_order,
            1,
            vec![(down_idx, 1)],
            ctx.n_anticipated,
            ctx.k_max,
            ctx.anticipated_lead_stages.clone(),
            &vec![0; ctx.n_hydros],
        );
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        let _ = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
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
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);

        // Block-dependent expression with block_id = None expands to one generic
        // row per block, so the constraint participates as `n_blks` rows.
        let n_blks = layout.n_blks;
        assert_eq!(
            layout.rows.n_generic_rows, n_blks,
            "block-dependent pumping constraint must expand to one row per block"
        );

        let mut col_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); layout.num_cols];
        let mut col_upper = vec![f64::INFINITY; layout.num_cols];
        let mut objective = vec![0.0_f64; layout.num_cols];
        let mut row_lower = vec![f64::NEG_INFINITY; layout.rows.num_rows];
        let mut row_upper = vec![f64::INFINITY; layout.rows.num_rows];
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
            let row = layout.rows.row_generic_start + blk;
            let col = layout.equipment.col_pumping_start + blk;
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

    // ── Filling-cascade test helpers (shared by the filling-row tests below) ─────

    use cobre_core::entities::hydro::FillingConfig;

    const RET_START_STAGE_ID: i32 = 2;
    const RET_ENTRY_STAGE_ID: i32 = 4;
    const RET_FILLING_ID: i32 = 3; // start <= 3 < entry  -> Filling
    const RET_OPERATING_ID: i32 = 4; // 4 >= entry         -> Operating

    /// A cascade hydro with caller-chosen `downstream_id`, `entry_stage_id`, and
    /// optional `FillingConfig`, so an upstream→downstream cascade with a filling
    /// downstream can be built. Constant productivity (no FPHA columns).
    fn ret_hydro(id: i32, downstream: Option<i32>, entry: Option<i32>, filling: bool) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            downstream_id: downstream.map(EntityId),
            travel_time_hours: None,
            entry_stage_id: entry,
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
            filling: filling.then_some(FillingConfig {
                start_stage_id: RET_START_STAGE_ID,
                filling_min_rate_m3s: 0.0,
            }),
            penalties: zero_hydro_penalties(),
        }
    }

    /// Sum the CSC values stored at `(col, row)`.
    fn csc_at(csc: &(Vec<i32>, Vec<i32>, Vec<f64>), col: usize, row: usize) -> f64 {
        let start = usize::try_from(csc.0[col]).unwrap();
        let end = usize::try_from(csc.0[col + 1]).unwrap();
        csc.1[start..end]
            .iter()
            .zip(&csc.2[start..end])
            .filter(|(r, _)| usize::try_from(**r).is_ok_and(|r| r == row))
            .map(|(_, &v)| v)
            .sum()
    }

    // ── PreFilling upstream of a Filling downstream (balance-row routing) ────────

    /// Like [`ret_hydro`] (filling) but with a caller-chosen filling
    /// `start_stage_id`, so an upstream filling hydro can begin its `PreFilling`
    /// phase LATER than a downstream filling hydro — the topology where the
    /// downstream is Filling while the upstream is still `PreFilling` at the same
    /// stage.
    fn ret_hydro_start(
        id: i32,
        downstream: Option<i32>,
        entry: Option<i32>,
        start_stage_id: i32,
    ) -> Hydro {
        let mut h = ret_hydro(id, downstream, entry, false);
        h.filling = Some(FillingConfig {
            start_stage_id,
            filling_min_rate_m3s: 0.0,
        });
        h
    }

    struct PfuOffsets {
        zeta: f64,
        z_u: usize,
        water_row_u: usize,
        water_row_d: usize,
        z_inflow_row_u: usize,
        filling_target_row_d: usize,
        n_target_rows: usize,
        storage_in_u: usize,
    }

    /// Count the nonzero entries stored in CSC column `col`.
    fn csc_col_nnz(csc: &(Vec<i32>, Vec<i32>, Vec<f64>), col: usize) -> usize {
        let start = usize::try_from(csc.0[col]).unwrap();
        let end = usize::try_from(csc.0[col + 1]).unwrap();
        csc.2[start..end].iter().filter(|&&v| v != 0.0).count()
    }

    /// Build `U(H1) → D(H2)` where BOTH are filling but the upstream U commissions
    /// LATER than the downstream D, evaluated at stage 2 where D is Filling
    /// (`start_D = 2 ≤ 2 < entry_D = 4`) and U is still `PreFilling`
    /// (`2 < start_U = 3`).
    #[allow(clippy::type_complexity)]
    fn build_prefilling_upstream_of_filling_case() -> ((Vec<i32>, Vec<i32>, Vec<f64>), PfuOffsets) {
        let stage_id = 2;
        let fixtures = PumpFixtures::new(
            vec![
                ret_hydro_start(1, Some(2), Some(5), 3), // U: PreFilling at 0,1,2; Filling at 3,4
                ret_hydro_start(2, None, Some(4), 2),    // D: Filling at 2,3; downstream of U
            ],
            Vec::new(),
        );
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let u_idx = fixtures.hydro_pos[&EntityId(1)];
        let d_idx = fixtures.hydro_pos[&EntityId(2)];
        // D is the only Filling hydro at this stage, so its σ_fill target row is the
        // single `filling_target` row — used below to prove routed inflow does NOT
        // land on it.
        let d_target_local = layout
            .filling
            .filling_target_hydro_indices
            .iter()
            .position(|&h| h.get() == d_idx)
            .expect("D is Filling, so it carries a σ_fill target row");
        let offsets = PfuOffsets {
            zeta: layout.zeta,
            z_u: layout.col_z_inflow_start() + u_idx,
            water_row_u: layout.rows.water_balance.start + u_idx,
            water_row_d: layout.rows.water_balance.start + d_idx,
            z_inflow_row_u: layout.rows.z_inflow_row_start + u_idx,
            filling_target_row_d: layout.filling.row_filling_target_start + d_target_local,
            n_target_rows: layout.filling.filling_target_hydro_indices.len(),
            storage_in_u: layout.col_storage_in_start() + u_idx,
        };
        (csc, offsets)
    }

    /// Regression: when a `PreFilling` hydro U is the direct upstream of a Filling
    /// hydro D at the same stage, U's routed natural inflow (`z_U` at `−ζ`) lands on
    /// D's water-balance row and on NO OTHER constraint row. Filling is uncapped in
    /// the volume-target model — there is no impound-cap row — so the routed inflow
    /// has only the balance row to receive it. The forbidden alternative (the v1
    /// cap-row port) routed a SECOND `z_U` push onto a retention row; removing the
    /// balance-row push instead would strand U's water (a correctness regression), so
    /// this test pins both: routed onto the balance row, and onto no other row.
    ///
    /// The `z_U` column carries exactly TWO nonzero entries: its `+1.0` on U's own
    /// z-inflow DEFINITION row (untouched by the short-circuit) and the routed `−ζ`
    /// on D's water-balance row. A cap-row port would add a third entry.
    #[test]
    fn prefilling_upstream_inflow_lands_on_balance_row_only() {
        let (csc, off) = build_prefilling_upstream_of_filling_case();
        assert_eq!(
            off.n_target_rows, 1,
            "exactly one σ_fill target row (only D is Filling)"
        );

        // U's realized inflow z_U is routed onto D's standard water-balance row at −ζ.
        assert_eq!(
            csc_at(&csc, off.z_u, off.water_row_d),
            -off.zeta,
            "z_U is routed onto D's water-balance row at −ζ (short-circuit target)"
        );
        // …and on NO OTHER constraint row beyond its own z-inflow definition row:
        // exactly two nonzero entries (def row +1.0, routed balance-row −ζ). A
        // retention/cap-row port would push a third entry here.
        assert_eq!(
            csc_at(&csc, off.z_u, off.z_inflow_row_u),
            1.0,
            "z_U keeps its +1.0 on U's own z-inflow definition row"
        );
        assert_eq!(
            csc_col_nnz(&csc, off.z_u),
            2,
            "z_U has exactly two nonzero CSC entries (def row + balance-row push, no cap-row port)"
        );
        // Specifically, z_U is absent from D's σ_fill target row (the σ_fill row
        // couples v_h + slack, never the routed inflow) and from U's own frozen
        // water-balance row.
        assert_eq!(
            csc_at(&csc, off.z_u, off.filling_target_row_d),
            0.0,
            "z_U does not land on D's σ_fill target row"
        );
        assert_eq!(
            csc_at(&csc, off.z_u, off.water_row_u),
            0.0,
            "z_U is relocated off U's frozen-identity water-balance row (PreFilling)"
        );
        assert_eq!(
            csc_at(&csc, off.storage_in_u, off.water_row_u),
            -1.0,
            "U's frozen-identity row keeps its incoming-storage −1 entry"
        );
    }

    // ── σ_fill terminal target row + column ──────────────────────────────────

    // The terminal Filling stage of the `ret_hydro` window: entry − 1 = 3.
    const TARGET_TERMINAL_ID: i32 = RET_ENTRY_STAGE_ID - 1;
    // A non-zero resolved dead volume so the σ_fill row RHS is observable.
    const TARGET_MIN_STORAGE_HM3: f64 = 37.5;

    struct TargetOffsets {
        n_target_rows: usize,
        target_row: usize,
        sigma_fill_col: usize,
        v_h_col: usize,
        num_rows: usize,
        num_cols: usize,
    }

    /// Build the `H1 → H2` cascade (H2 the filling hydro, `entry = 4`) at the given
    /// `stage_id`, with H2's resolved per-stage `min_storage_hm3` set to
    /// `TARGET_MIN_STORAGE_HM3`. Returns the assembled CSC triple, the
    /// `(row_lower, row_upper)` vectors, and the `σ_fill` offsets the assertions read.
    #[allow(clippy::type_complexity)]
    fn build_target_case(
        stage_id: i32,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        TargetOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures.bounds.hydro_bounds_mut(h2_idx, 0).min_storage_hm3 = TARGET_MIN_STORAGE_HM3;
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let offsets = TargetOffsets {
            n_target_rows: layout.filling.filling_target_hydro_indices.len(),
            target_row: layout.filling.row_filling_target_start,
            sigma_fill_col: layout.filling.col_filling_target_start,
            // The outgoing storage column v_h is the dense system index h2_idx.
            v_h_col: h2_idx,
            num_rows: layout.rows.num_rows,
            num_cols: layout.num_cols,
        };
        (csc, row_lower, row_upper, offsets)
    }

    /// At the last Filling stage (id == entry − 1 == 3) the `σ_fill` row exists with
    /// exactly `+1` on the outgoing storage column `v_h`, `+1` on the `σ_fill` slack
    /// column, `≥` sense, and RHS `V_target[3] == min_storage_hm3` (the backward-fold
    /// anchor; rate is 0 here so the flat trajectory pins every floor to the dead
    /// volume). Exactly one such row + column (the single filling hydro H2).
    #[test]
    fn sigma_fill_row_columns_coefficients_and_rhs_at_last_filling_stage() {
        let (csc, row_lower, row_upper, off) = build_target_case(TARGET_TERMINAL_ID);
        assert_eq!(
            off.n_target_rows, 1,
            "exactly one σ_fill row (H2 last Filling)"
        );
        let row = off.target_row;
        assert_eq!(
            csc_at(&csc, off.v_h_col, row),
            1.0,
            "+1 on the outgoing storage column v_h"
        );
        assert_eq!(
            csc_at(&csc, off.sigma_fill_col, row),
            1.0,
            "+1 on the σ_fill slack column"
        );
        assert_eq!(
            row_lower[row], TARGET_MIN_STORAGE_HM3,
            "σ_fill row_lower == V_target[L] == min_storage_hm3 (≥ RHS at the anchor)"
        );
        assert_eq!(
            row_upper[row],
            f64::INFINITY,
            "σ_fill row is a ≥ inequality (upper = +∞)"
        );
    }

    /// No `σ_fill` row or column is emitted OFF the Filling phase — `PreFilling`
    /// (id 1, before start = 2) or `Operating` (id 4, at entry). Per-stage Filling
    /// membership; the v1 terminal-only rule is gone.
    ///
    /// At `PreFilling` the `σ^{v-}` family is also empty, so the empty `σ_fill`
    /// block's cursor coincides with `num_rows`/`num_cols`. At `Operating` the
    /// `σ^{v-}` family legitimately adds exactly one row + column for the single
    /// filling hydro AFTER the empty `σ_fill` block, so the `σ_fill` cursor sits one
    /// short of `num_rows`/`num_cols` — the `σ_fill` block is still empty
    /// (`n_target_rows == 0`), it is simply no longer the last family.
    #[test]
    fn sigma_fill_absent_off_filling_phase() {
        // id 1 is PreFilling (start = 2 > 0, stage_id < start); id 4 is Operating.
        for stage_id in [RET_START_STAGE_ID - 1, RET_OPERATING_ID] {
            let (_csc, _rl, _ru, off) = build_target_case(stage_id);
            assert_eq!(
                off.n_target_rows, 0,
                "no σ_fill row at non-Filling id {stage_id}"
            );
            // σ^{v-} adds one row + column at Operating (the single filling hydro),
            // none at PreFilling.
            let sigma_minus_width = usize::from(stage_id == RET_OPERATING_ID);
            assert_eq!(
                off.target_row + sigma_minus_width,
                off.num_rows,
                "empty σ_fill row block at id {stage_id} (σ^{{v-}} occupies the tail in Operating)"
            );
            assert_eq!(
                off.sigma_fill_col + sigma_minus_width,
                off.num_cols,
                "empty σ_fill column block at id {stage_id} (σ^{{v-}} occupies the tail in Operating)"
            );
        }
    }

    /// At an EARLY Filling stage (id 2 = start, with the last Filling stage at id 3)
    /// the `σ_fill` row RHS is the backward-folded `V_target[2] == min_storage −
    /// ζ_3·rate_3`, strictly BELOW the dead volume — NOT `min_storage` at every
    /// stage. This is the per-stage trajectory: the reservoir is only required to
    /// hold the running minimum by stage 2, with the full dead volume due at stage 3.
    /// Uses the AC schedule (`min_storage = 60`, per-stage ζ = 2.592, rate = 5) so
    /// `V_target[3] = 60`, `V_target[2] = 60 − 2.592·5 = 47.04`.
    #[test]
    fn sigma_fill_row_rhs_is_backward_anchored_v_target_at_early_filling_stage() {
        // AC values. The fixture's `build_filling_v_target` maps ids 2 and 3 to
        // stage_idx 0; set H2's resolved dead volume and per-stage fill rate at
        // stage_idx 0 so the fold reads them.
        const AC_MIN_STORAGE: f64 = 60.0;
        const AC_RATE_M3S: f64 = 5.0;
        // ζ = total_hours · M3S_TO_HM3 = 720 · 0.0036 = 2.592 (the AC ζ).
        const AC_TOTAL_HOURS: f64 = 720.0;

        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures.bounds.hydro_bounds_mut(h2_idx, 0).min_storage_hm3 = AC_MIN_STORAGE;
        fixtures
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .filling_min_rate_m3s = AC_RATE_M3S;
        // The fixture-default total_hours_per_stage is 744; rebuild the ctx's
        // V_target map with the AC ζ (720 h → ζ = 2.592) so the fold matches the AC.
        let ctx = TemplateBuildCtx {
            filling_v_target: super::super::template::build_filling_v_target(
                &fixtures.hydros,
                &fixtures.bounds,
                &[AC_TOTAL_HOURS],
                &(0..=8_i32).map(|id| (id, 0_usize)).collect(),
            ),
            ..fixtures.make_ctx()
        };

        // V_target[3] (last Filling stage) == min_storage.
        let v_target_last = ctx.filling_v_target[&(h2_idx, RET_FILLING_ID)];
        assert!(
            (v_target_last - AC_MIN_STORAGE).abs() < 1e-9,
            "V_target[3] == min_storage (anchor), got {v_target_last}"
        );

        // V_target[2] (early Filling stage) == 60 − 2.592·5 == 47.04.
        let v_target_early = ctx.filling_v_target[&(h2_idx, RET_START_STAGE_ID)];
        let expected_early = AC_MIN_STORAGE - AC_TOTAL_HOURS * M3S_TO_HM3 * AC_RATE_M3S;
        assert!(
            (v_target_early - 47.04).abs() < 1e-9,
            "V_target[2] == 47.04 (backward fold), got {v_target_early}"
        );
        assert!(
            (v_target_early - expected_early).abs() < 1e-9,
            "V_target[2] == min_storage − ζ·rate, got {v_target_early} vs {expected_early}"
        );
        assert!(
            v_target_early < v_target_last,
            "early floor strictly below the dead-volume anchor (per-stage trajectory)"
        );

        // The row RHS at id 2 reads that V_target[2], not min_storage.
        let stage = two_block_stage(usize::try_from(RET_START_STAGE_ID).unwrap(), [360.0, 360.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (row_lower, _row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let row = layout.filling.row_filling_target_start;
        assert_eq!(
            layout.filling.filling_target_hydro_indices.len(),
            1,
            "one σ_fill row at the early Filling stage (id 2)"
        );
        assert!(
            (row_lower[row] - 47.04).abs() < 1e-9,
            "σ_fill row_lower == V_target[2] == 47.04, got {}",
            row_lower[row]
        );
    }

    /// A non-filling system (no `FillingConfig` on any hydro) emits NO `σ_fill` row
    /// or column, and its `num_rows`/`num_cols` are unchanged at the would-be
    /// terminal stage — the parity-neutrality contract.
    #[test]
    fn non_filling_system_emits_no_sigma_fill() {
        let control = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, None, false),
            ],
            Vec::new(),
        );
        let ctx = control.make_ctx();
        let stage = two_block_stage(usize::try_from(TARGET_TERMINAL_ID).unwrap(), [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        assert_eq!(
            layout.filling.filling_target_hydro_indices.len(),
            0,
            "control system has no filling hydro ⇒ no σ_fill row/column"
        );
        // The σ_fill row/column cursors degenerate to the structural bounds.
        assert_eq!(
            layout.filling.row_filling_target_start,
            layout.rows.num_rows
        );
        assert_eq!(layout.filling.col_filling_target_start, layout.num_cols);
    }

    /// Cut-validity guard (§4 trap 3): the `σ_fill` soft row couples to the storage
    /// state through the constraint matrix, so LP duality folds its dual into the
    /// SINGLE reduced cost of the incoming-storage column the cut already reads as
    /// `rc / col_scale`. The `σ_fill` builder code therefore must NEVER reference the
    /// dual-extraction entry point — doing so would signal a hand-combination of the
    /// soft-row dual that double-counts the floor and corrupts the cut. This test
    /// reads every `lp/builder/` source file and asserts none mentions that symbol,
    /// so a future "simplification" that wires the soft dual in by hand fails here.
    /// The dual-extraction function itself lives in
    /// `training/backward/duals_extraction.rs`, untouched by this family.
    ///
    /// The forbidden symbol is assembled from chars so this guard's own source text
    /// does not contain the literal (which would make the test flag itself).
    #[test]
    fn lp_builder_never_references_dual_extraction() {
        use std::path::Path;
        // The dual-extraction symbol, assembled from fragments so the needle is
        // absent from this file's own bytes (else the guard would flag itself).
        let needle: String = ["extract", "_duals", "_from", "_view"].concat();
        let builder_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lp/builder");
        assert!(
            builder_dir.is_dir(),
            "lp/builder source dir must exist at {}",
            builder_dir.display()
        );
        let mut offenders: Vec<String> = Vec::new();
        for entry in std::fs::read_dir(&builder_dir).expect("read lp/builder dir") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let src = std::fs::read_to_string(&path).expect("read builder source file");
            if src.contains(&needle) {
                offenders.push(path.display().to_string());
            }
        }
        assert!(
            offenders.is_empty(),
            "lp/builder must not reference the dual-extraction symbol (the σ_fill \
             soft dual is folded into the storage column reduced cost by LP duality, \
             never hand-combined); offending files: {offenders:?}"
        );
    }

    /// The `σ_fill` row lands STRICTLY BELOW `num_rows` (inside the structural
    /// region, ahead of the appended cut rows that start at `num_rows`), and the
    /// `σ_fill` column strictly below `num_cols`. A `σ_fill` row at index `>= num_rows`
    /// would alias a cut row and corrupt slot-identity warm-start reconstruction.
    #[test]
    fn sigma_fill_row_below_num_rows_and_col_below_num_cols() {
        let (_csc, _rl, _ru, off) = build_target_case(TARGET_TERMINAL_ID);
        assert!(off.n_target_rows > 0, "this case has a σ_fill row");
        assert!(
            off.target_row + off.n_target_rows <= off.num_rows,
            "σ_fill rows [{}, {}) must lie within [0, num_rows={})",
            off.target_row,
            off.target_row + off.n_target_rows,
            off.num_rows
        );
        assert!(
            off.sigma_fill_col + off.n_target_rows <= off.num_cols,
            "σ_fill cols [{}, {}) must lie within [0, num_cols={})",
            off.sigma_fill_col,
            off.sigma_fill_col + off.n_target_rows,
            off.num_cols
        );
    }

    // ── σ^{v-} operating-floor row + column ──────────────────────────────────

    struct FloorOffsets {
        n_floor_rows: usize,
        floor_row: usize,
        sigma_minus_col: usize,
        v_h_col: usize,
        num_rows: usize,
        num_cols: usize,
    }

    /// Build the `H1 → H2` cascade (H2 the filling hydro, `entry = 4`) at the given
    /// `stage_id`, with H2's resolved per-stage `min_storage_hm3` set to
    /// `TARGET_MIN_STORAGE_HM3`. Returns the assembled CSC triple, the
    /// `(row_lower, row_upper)` vectors, and the `σ^{v-}` offsets the assertions
    /// read. Mirrors `build_target_case` but reads the `filled_min_storage_floor` family.
    #[allow(clippy::type_complexity)]
    fn build_floor_case(
        stage_id: i32,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        FloorOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures.bounds.hydro_bounds_mut(h2_idx, 0).min_storage_hm3 = TARGET_MIN_STORAGE_HM3;
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let offsets = FloorOffsets {
            n_floor_rows: layout.filling.filled_min_storage_floor_hydro_indices.len(),
            floor_row: layout.filling.row_filled_min_storage_floor_start,
            sigma_minus_col: layout.filling.col_filled_min_storage_floor_start,
            // The outgoing storage column v_h is the dense system index h2_idx.
            v_h_col: h2_idx,
            num_rows: layout.rows.num_rows,
            num_cols: layout.num_cols,
        };
        (csc, row_lower, row_upper, offsets)
    }

    /// At an Operating stage (id == entry == 4) the `σ^{v-}` row exists with exactly
    /// `+1` on the outgoing storage column `v_h`, `+1` on the `σ^{v-}` slack column,
    /// `≥` sense, and RHS `min_storage_hm3`. Exactly one such row + column (the
    /// single filling hydro H2). Same shape as the `σ_fill` row, different stage
    /// scope.
    #[test]
    fn sigma_minus_row_columns_coefficients_and_rhs_in_operating() {
        let (csc, row_lower, row_upper, off) = build_floor_case(RET_OPERATING_ID);
        assert_eq!(
            off.n_floor_rows, 1,
            "exactly one σ^{{v-}} row (H2 operating)"
        );
        let row = off.floor_row;
        assert_eq!(
            csc_at(&csc, off.v_h_col, row),
            1.0,
            "+1 on the outgoing storage column v_h"
        );
        assert_eq!(
            csc_at(&csc, off.sigma_minus_col, row),
            1.0,
            "+1 on the σ^{{v-}} slack column"
        );
        assert_eq!(
            row_lower[row], TARGET_MIN_STORAGE_HM3,
            "σ^{{v-}} row_lower == min_storage_hm3 (≥ RHS)"
        );
        assert_eq!(
            row_upper[row],
            f64::INFINITY,
            "σ^{{v-}} row is a ≥ inequality (upper = +∞)"
        );
    }

    /// No `σ^{v-}` row or column is emitted at a non-operating stage of a filling
    /// hydro — `PreFilling` (id 0) or a Filling stage (id 3). Operating-only.
    #[test]
    fn sigma_minus_absent_off_operating_stage() {
        for stage_id in [RET_PREFILLING_ID, TARGET_TERMINAL_ID] {
            let (_csc, _rl, _ru, off) = build_floor_case(stage_id);
            assert_eq!(
                off.n_floor_rows, 0,
                "no σ^{{v-}} row at id {stage_id} (only Operating carries it)"
            );
            assert_eq!(
                off.floor_row, off.num_rows,
                "empty σ^{{v-}} row block: cursor coincides with num_rows at id {stage_id}"
            );
            assert_eq!(
                off.sigma_minus_col, off.num_cols,
                "empty σ^{{v-}} column block: cursor coincides with num_cols at id {stage_id}"
            );
        }
    }

    /// At the terminal `Filling` stage (id == entry − 1 == 3) a filling hydro has
    /// the `σ_fill` family but NOT `σ^{v-}`; in `Operating` (id == entry == 4) it has
    /// `σ^{v-}` but NOT `σ_fill` — the two families are mutually exclusive by stage
    /// and must not be conflated.
    #[test]
    fn sigma_minus_and_sigma_fill_mutually_exclusive_by_stage() {
        let (_csc_t, _rl_t, _ru_t, target_at_terminal) = build_target_case(TARGET_TERMINAL_ID);
        let (_csc_tf, _rl_tf, _ru_tf, floor_at_terminal) = build_floor_case(TARGET_TERMINAL_ID);
        assert_eq!(
            target_at_terminal.n_target_rows, 1,
            "σ_fill present at terminal Filling stage"
        );
        assert_eq!(
            floor_at_terminal.n_floor_rows, 0,
            "σ^{{v-}} absent at terminal Filling stage"
        );

        let (_csc_o, _rl_o, _ru_o, target_in_op) = build_target_case(RET_OPERATING_ID);
        let (_csc_of, _rl_of, _ru_of, floor_in_op) = build_floor_case(RET_OPERATING_ID);
        assert_eq!(target_in_op.n_target_rows, 0, "σ_fill absent in Operating");
        assert_eq!(floor_in_op.n_floor_rows, 1, "σ^{{v-}} present in Operating");
    }

    /// A non-filling system (no `FillingConfig` on any hydro) emits NO `σ^{v-}` row
    /// or column, and its `num_rows`/`num_cols` are unchanged at the would-be
    /// Operating stage — the parity-neutrality contract.
    #[test]
    fn non_filling_system_emits_no_sigma_minus() {
        let control = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, None, false),
            ],
            Vec::new(),
        );
        let ctx = control.make_ctx();
        let stage = two_block_stage(usize::try_from(RET_OPERATING_ID).unwrap(), [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        assert_eq!(
            layout.filling.filled_min_storage_floor_hydro_indices.len(),
            0,
            "control system has no filling hydro ⇒ no σ^{{v-}} row/column"
        );
        // The σ^{v-} row/column cursors degenerate to the structural bounds.
        assert_eq!(
            layout.filling.row_filled_min_storage_floor_start,
            layout.rows.num_rows
        );
        assert_eq!(
            layout.filling.col_filled_min_storage_floor_start,
            layout.num_cols
        );
    }

    /// The `σ^{v-}` row lands STRICTLY BELOW `num_rows` (inside the structural
    /// region, ahead of the appended cut rows that start at `num_rows`), and the
    /// `σ^{v-}` column strictly below `num_cols`. A `σ^{v-}` row at index
    /// `>= num_rows` would alias a cut row and corrupt slot-identity warm-start
    /// reconstruction.
    #[test]
    fn sigma_minus_row_below_num_rows_and_col_below_num_cols() {
        let (_csc, _rl, _ru, off) = build_floor_case(RET_OPERATING_ID);
        assert!(off.n_floor_rows > 0, "this case has a σ^{{v-}} row");
        assert!(
            off.floor_row + off.n_floor_rows <= off.num_rows,
            "σ^{{v-}} rows [{}, {}) must lie within [0, num_rows={})",
            off.floor_row,
            off.floor_row + off.n_floor_rows,
            off.num_rows
        );
        assert!(
            off.sigma_minus_col + off.n_floor_rows <= off.num_cols,
            "σ^{{v-}} cols [{}, {}) must lie within [0, num_cols={})",
            off.sigma_minus_col,
            off.sigma_minus_col + off.n_floor_rows,
            off.num_cols
        );
    }

    // ── PreFilling cascade short-circuit ─────────────────────────────────────

    // `RET_START_STAGE_ID = 2`, so a stage with id `< 2` is PreFilling for a
    // filling hydro built by `ret_hydro`. Stage id 0 is the canonical PreFilling
    // probe; the boundary stage id 2 is Filling (governed by the retention tests
    // above), confirming the short-circuit does NOT fire there.
    const RET_PREFILLING_ID: i32 = 0;

    /// Resolved offsets for the H1→H2→H3 routed short-circuit probe.
    struct ScOffsets {
        zeta: f64,
        n_blks: usize,
        h2_idx: usize,
        water_row_h2: usize,
        water_row_h3: usize,
        col_storage_in_h2: usize,
        z_h2: usize,
        h1_turbine: Vec<usize>,
        h1_spillage: Vec<usize>,
        h2_turbine: Vec<usize>,
        h2_spillage: Vec<usize>,
    }

    /// Build the cascade `H1(id 1) → H2(id 2, filling) → H3(id 3)` at `stage_id`,
    /// with H2's resolved per-stage withdrawal set to `withdrawal_h`. Returns the
    /// assembled CSC, the `(row_lower, row_upper)` vectors, and the offsets the
    /// short-circuit assertions read. H2 is the mid-cascade filling hydro, so the
    /// short-circuit routes to a REAL downstream (H3), not a sink.
    #[allow(clippy::type_complexity)]
    fn build_shortcircuit_case(
        stage_id: i32,
        withdrawal_h: f64,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        ScOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, Some(3), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(3, None, None, false),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = withdrawal_h;
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let h3_idx = fixtures.hydro_pos[&EntityId(3)];
        let offsets = ScOffsets {
            zeta: layout.zeta,
            n_blks: layout.n_blks,
            h2_idx,
            water_row_h2: layout.rows.water_balance.start + h2_idx,
            water_row_h3: layout.rows.water_balance.start + h3_idx,
            col_storage_in_h2: layout.col_storage_in_start() + h2_idx,
            z_h2: layout.col_z_inflow_start() + h2_idx,
            h1_turbine: (0..layout.n_blks)
                .map(|blk| layout.turbine_col(HydroSys::new(h1_idx), blk))
                .collect(),
            h1_spillage: (0..layout.n_blks)
                .map(|blk| layout.spillage_col(HydroSys::new(h1_idx), blk))
                .collect(),
            h2_turbine: (0..layout.n_blks)
                .map(|blk| layout.turbine_col(HydroSys::new(h2_idx), blk))
                .collect(),
            h2_spillage: (0..layout.n_blks)
                .map(|blk| layout.spillage_col(HydroSys::new(h2_idx), blk))
                .collect(),
        };
        (csc, row_lower, row_upper, offsets)
    }

    /// At a `PreFilling` stage the absent hydro `H2`'s four water interactions land
    /// on the DOWNSTREAM `H3`'s balance row in real `ζ`/`τ_k` coefficients:
    /// `H2`'s incremental inflow (`−ζ` on `z_{H2}`), `H2`'s upstream `H1` releases
    /// (`−τ` on `H1` turbine/spillage), and `H2`'s withdrawal demand (`−ζ·withdrawal`
    /// on `H3`'s RHS). `H2`'s evaporation is NOT transferred (no evap column exists
    /// — `PreFilling` is excluded upstream). The same columns carry NOTHING extra on
    /// `H2`'s own (frozen) row.
    #[test]
    fn prefilling_routes_inflow_upstream_releases_and_withdrawal_to_downstream() {
        let withdrawal_h = 17.5_f64;
        let (csc, _rl, row_upper, off) = build_shortcircuit_case(RET_PREFILLING_ID, withdrawal_h);
        let row_d = off.water_row_h3;
        let row_h = off.water_row_h2;

        // (1) Incremental inflow: z_{H2} carries −ζ on H3's row (relocated), and
        // NOTHING on H2's own frozen row.
        assert_eq!(
            csc_at(&csc, off.z_h2, row_d),
            -off.zeta,
            "z_{{H2}} carries −ζ on H3's water row (incremental inflow re-routed)"
        );
        assert_eq!(
            csc_at(&csc, off.z_h2, row_h),
            0.0,
            "z_{{H2}} carries nothing on H2's own frozen-identity row"
        );

        // (2) Upstream H1 releases re-routed upstream(H2)→H3 with −τ_h per block.
        // H1 is NOT a standard upstream of H3 (the standard cascade edge is
        // H1→H2→H3), so a nonzero −τ on H1's columns at H3's row can ONLY come from
        // the short-circuit re-route. (H2's own release columns DO appear on H3's
        // row via H3's standard upstream loop — H2 is structurally upstream of H3 —
        // but H2's turbine/spillage are pinned `[0,0]` in PreFilling, so that
        // standard coefficient multiplies a zero column and is harmless. The
        // re-route is what makes H1's water reach H3.)
        for blk in 0..off.n_blks {
            let tau_h = [300.0_f64, 444.0][blk] * M3S_TO_HM3;
            assert_eq!(
                csc_at(&csc, off.h1_turbine[blk], row_d),
                -tau_h,
                "blk {blk}: H1 turbine carries −τ_h on H3's row (cascade edge \
                 H1→H2→H3 re-routed to H1→H3 — H1 is NOT a standard upstream of H3)"
            );
            assert_eq!(
                csc_at(&csc, off.h1_spillage[blk], row_d),
                -tau_h,
                "blk {blk}: H1 spillage carries −τ_h on H3's row"
            );
        }

        // (3) Withdrawal demand transferred to H3's RHS: H3's row_upper drops by
        // ζ·withdrawal_h versus a no-withdrawal build (its own withdrawal is 0).
        let (_csc0, _rl0, row_upper0, off0) = build_shortcircuit_case(RET_PREFILLING_ID, 0.0);
        let delta = off.zeta * withdrawal_h;
        assert_eq!(
            row_upper0[off0.water_row_h3] - row_upper[row_d],
            delta,
            "H3's RHS drops by ζ·withdrawal_{{H2}} (the transferred withdrawal demand)"
        );
    }

    /// `H2`'s own (frozen) row is exactly `v_{H2} − v_{H2,in} = 0`: `+1.0` on its
    /// outgoing-storage column, `−1.0` on its incoming-storage column, and NOTHING
    /// else (no inflow/upstream/AR-lag/withdrawal/evaporation), with RHS 0. The
    /// storage column keeps its dense system index (the absent reservoir's column
    /// is never omitted or relocated — §4 trap 2).
    #[test]
    fn prefilling_h_row_is_frozen_identity_with_dense_storage_column() {
        let (csc, row_lower, row_upper, off) = build_shortcircuit_case(RET_PREFILLING_ID, 9.0);
        let row_h = off.water_row_h2;

        // Outgoing storage column index == H2's system hydro index (dense, unchanged
        // from an operating build — the storage column is never relocated).
        assert_eq!(
            csc_at(&csc, off.h2_idx, row_h),
            1.0,
            "outgoing storage v_{{H2}} carries +1.0 on the frozen-identity row, \
             at the dense column index == H2's system hydro index"
        );
        assert_eq!(
            csc_at(&csc, off.col_storage_in_h2, row_h),
            -1.0,
            "incoming storage v_{{H2,in}} carries −1.0 on the frozen-identity row"
        );

        // Nothing else on H2's row: own releases, z-coupling, withdrawal slacks.
        for blk in 0..off.n_blks {
            assert_eq!(
                csc_at(&csc, off.h2_turbine[blk], row_h),
                0.0,
                "blk {blk}: own turbine absent from the frozen row"
            );
            assert_eq!(
                csc_at(&csc, off.h2_spillage[blk], row_h),
                0.0,
                "blk {blk}: own spillage absent from the frozen row"
            );
            assert_eq!(
                csc_at(&csc, off.h1_turbine[blk], row_h),
                0.0,
                "blk {blk}: upstream H1 turbine absent from H2's frozen row \
                 (re-routed to H3, not left on H2 — §4 trap 4)"
            );
        }
        assert_eq!(
            csc_at(&csc, off.z_h2, row_h),
            0.0,
            "z_{{H2}} absent from the frozen row"
        );

        // RHS is exactly 0 (NOT ζ·(base − withdrawal)).
        assert_eq!(row_lower[row_h], 0.0, "frozen-identity row_lower == 0");
        assert_eq!(row_upper[row_h], 0.0, "frozen-identity row_upper == 0");
    }

    /// Sink case: a `PreFilling` hydro with `downstream_id == None` drops its water
    /// from the system (exactly as a terminal hydro's outflow today). No downstream
    /// row receives the inflow/upstream/withdrawal, and `H2`'s own row is still the
    /// frozen identity. Builds `H1 → H2(filling, terminal)`.
    #[test]
    fn prefilling_sink_drops_water_and_keeps_frozen_identity() {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = 13.0;
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(usize::try_from(RET_PREFILLING_ID).unwrap(), [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let row_h = layout.rows.water_balance.start + h2_idx;
        let z_h2 = layout.col_z_inflow_start() + h2_idx;

        // Frozen identity intact on H2's own row.
        assert_eq!(csc_at(&csc, h2_idx, row_h), 1.0, "v_{{H2}} +1.0");
        assert_eq!(
            csc_at(&csc, layout.col_storage_in_start() + h2_idx, row_h),
            -1.0,
            "v_{{H2,in}} −1.0"
        );
        assert_eq!(
            row_lower[row_h], 0.0,
            "frozen RHS 0 (no withdrawal folded in)"
        );
        assert_eq!(row_upper[row_h], 0.0, "frozen RHS 0");

        // H2's water exits the system: z_{H2} appears on NO water-balance row, and
        // H1's releases appear only on H1's own row (no downstream to feed).
        for h in 0..layout.n_h {
            let r = layout.rows.water_balance.start + h;
            assert_eq!(
                csc_at(&csc, z_h2, r),
                0.0,
                "z_{{H2}} must not land on any water row in the sink case (row {r})"
            );
        }
        let row_h1 = layout.rows.water_balance.start + h1_idx;
        for blk in 0..layout.n_blks {
            let tau_h = [300.0_f64, 444.0][blk] * M3S_TO_HM3;
            // H1's own +τ on its own row is unchanged; it lands on NO other water row.
            assert_eq!(
                csc_at(&csc, layout.turbine_col(HydroSys::new(h1_idx), blk), row_h1),
                tau_h,
                "blk {blk}: H1's own turbine still carries +τ on its own row"
            );
            assert_eq!(
                csc_at(&csc, layout.turbine_col(HydroSys::new(h1_idx), blk), row_h),
                0.0,
                "blk {blk}: H1's turbine does not feed the absent sink H2"
            );
        }
    }

    /// Duals test (real solver): at a `PreFilling` stage the incoming-storage column's
    /// reduced cost is 0 — the §4 trap-4 contract `∂Q/∂v̂_h = 0` (a valid flat cut).
    /// With `v_{H2,in}` relocated out of `H2`'s row (the row is the frozen identity
    /// and `v_{H2}` is a dead variable), perturbing the pinned `v̂_{H2}` changes
    /// neither the reduced cost nor the objective. A botched relocation that left
    /// `H2`'s row coupled to upstream would give a nonzero reduced cost
    /// (`β_{H2}` stale-nonzero).
    #[test]
    fn prefilling_incoming_storage_reduced_cost_is_zero() {
        use cobre_solver::{ActiveSolver, SolverInterface};

        let fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, Some(3), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(3, None, None, false),
            ],
            Vec::new(),
        )
        .with_resolved_penalties();
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(usize::try_from(RET_PREFILLING_ID).unwrap(), [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let out = super::super::template::build_single_stage_template(&ctx, &state, &stage, 0);
        let template = out.template;

        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        let col_v_in = state
            .state_to_lp_incoming_column(StateDim::new(h2_idx))
            .get();

        // Solve twice with different pinned v̂_{H2}; assert the incoming-storage
        // reduced cost is 0 at both and the objective is unchanged. col_scale is
        // empty for this freshly built (unscaled) template, so the reduced cost is
        // already in original units (no /col_scale needed).
        let solve_at = |v_hat: f64| -> (f64, f64) {
            let mut solver = ActiveSolver::new().expect("ActiveSolver::new()");
            solver.load_model(&template);
            solver.set_col_bounds(&[col_v_in], &[v_hat], &[v_hat]);
            let view = solver.solve(None).expect("PreFilling LP must be feasible");
            (view.reduced_costs[col_v_in], view.objective)
        };

        let (rc_lo, obj_lo) = solve_at(10.0);
        let (rc_hi, obj_hi) = solve_at(70.0);

        assert!(
            rc_lo.abs() < 1e-9,
            "incoming-storage reduced cost must be 0 at v̂=10 (flat cut), got {rc_lo}"
        );
        assert!(
            rc_hi.abs() < 1e-9,
            "incoming-storage reduced cost must be 0 at v̂=70 (flat cut), got {rc_hi}"
        );
        assert!(
            (obj_lo - obj_hi).abs() < 1e-9,
            "objective must be invariant to v̂_{{H2}} (∂Q/∂v̂_h = 0): {obj_lo} vs {obj_hi}"
        );
    }

    /// Non-filling parity: building the SAME 3-hydro cascade with H2 NOT filling
    /// produces a bit-identical water-balance row for H2 (no short-circuit, no
    /// relocation) and identical `num_rows`. The `PreFilling` logic is a no-op for a
    /// hydro with no `FillingConfig`.
    #[test]
    fn non_filling_hydro_water_row_and_num_rows_bit_identical() {
        // Filling build at the PreFilling stage: H2 IS short-circuited.
        let (csc_f, rl_f, ru_f, off_f) = build_shortcircuit_case(RET_PREFILLING_ID, 0.0);

        // Control: same topology and stage, but H2 carries no filling ⇒ Operating
        // everywhere ⇒ standard balance row, no short-circuit.
        let control = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, Some(3), None, false),
                ret_hydro(3, None, None, false),
            ],
            Vec::new(),
        );
        let ctx = control.make_ctx();
        let stage = two_block_stage(usize::try_from(RET_PREFILLING_ID).unwrap(), [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (rl_c, ru_c) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc_c = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let h2_idx_c = control.hydro_pos[&EntityId(2)];
        let row_h2_c = layout.rows.water_balance.start + h2_idx_c;

        // num_rows identical (no extra structural rows from the short-circuit; it
        // only moves coefficients, never adds rows).
        assert_eq!(
            layout.rows.num_rows,
            ru_f.len(),
            "num_rows identical whether H2 is PreFilling or non-filling"
        );

        // H2's OWN water row differs between the builds (frozen vs standard) — that
        // is the short-circuit. The control's H2 row carries its own +τ releases;
        // assert at least the own-turbine coefficient is present in the control and
        // ABSENT in the filling build (the observable short-circuit).
        let row_h2_f = off_f.water_row_h2;
        for blk in 0..off_f.n_blks {
            let tau_h = [300.0_f64, 444.0][blk] * M3S_TO_HM3;
            assert_eq!(
                csc_at(
                    &csc_c,
                    layout.turbine_col(HydroSys::new(h2_idx_c), blk),
                    row_h2_c
                ),
                tau_h,
                "blk {blk}: control (non-filling) H2 carries +τ on its own row"
            );
            assert_eq!(
                csc_at(&csc_f, off_f.h2_turbine[blk], row_h2_f),
                0.0,
                "blk {blk}: filling (PreFilling) H2 row is frozen — no own +τ"
            );
        }

        // The non-filling control's balance row is a non-zero equality RHS
        // (ζ·base, here base=0 with no PAR ⇒ 0) and an equality (lower == upper).
        assert_eq!(
            rl_c[row_h2_c], ru_c[row_h2_c],
            "control H2 row is an equality"
        );
        // Sanity: the filling build's H2 row is the frozen identity (0).
        assert_eq!(rl_f[row_h2_f], 0.0, "filling H2 frozen RHS == 0");
        assert_eq!(ru_f[row_h2_f], 0.0, "filling H2 frozen RHS == 0");
    }

    // ── Commissioning-dormant NON-filling hydro (reuses the PreFilling path) ─────

    /// A NON-filling cascade hydro with a caller-chosen `entry_stage_id` and no
    /// `FillingConfig`: commissioning-dormant (`PreFilling`) before `entry`, then
    /// `Operating` from `entry` with no intervening `Filling`.
    fn ret_hydro_nonfilling(id: i32, downstream: Option<i32>, entry: Option<i32>) -> Hydro {
        ret_hydro(id, downstream, entry, false)
    }

    /// Build `H1 → H2(non-filling, entry) → H3` at `stage_id`, with H2's resolved
    /// withdrawal set to `withdrawal_h`. Mirrors [`build_shortcircuit_case`] but H2
    /// carries a commissioning window instead of a `FillingConfig`.
    #[allow(clippy::type_complexity)]
    fn build_nonfilling_shortcircuit_case(
        stage_id: i32,
        entry: i32,
        withdrawal_h: f64,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        ScOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro_nonfilling(1, Some(2), None),
                ret_hydro_nonfilling(2, Some(3), Some(entry)),
                ret_hydro_nonfilling(3, None, None),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = withdrawal_h;
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let h3_idx = fixtures.hydro_pos[&EntityId(3)];
        let offsets = ScOffsets {
            zeta: layout.zeta,
            n_blks: layout.n_blks,
            h2_idx,
            water_row_h2: layout.rows.water_balance.start + h2_idx,
            water_row_h3: layout.rows.water_balance.start + h3_idx,
            col_storage_in_h2: layout.col_storage_in_start() + h2_idx,
            z_h2: layout.col_z_inflow_start() + h2_idx,
            h1_turbine: (0..layout.n_blks)
                .map(|blk| layout.turbine_col(HydroSys::new(h1_idx), blk))
                .collect(),
            h1_spillage: (0..layout.n_blks)
                .map(|blk| layout.spillage_col(HydroSys::new(h1_idx), blk))
                .collect(),
            h2_turbine: (0..layout.n_blks)
                .map(|blk| layout.turbine_col(HydroSys::new(h2_idx), blk))
                .collect(),
            h2_spillage: (0..layout.n_blks)
                .map(|blk| layout.spillage_col(HydroSys::new(h2_idx), blk))
                .collect(),
        };
        (csc, row_lower, row_upper, offsets)
    }

    /// A commissioning-dormant non-filling hydro H2 (entry 4, evaluated at stage 0)
    /// is `PreFilling`: its own water row is the frozen identity `v_{H2} −
    /// v_{H2,in} = 0` (RHS 0) and its incremental inflow `z_{H2}` lands on the
    /// downstream H3's balance row at `−ζ` and nowhere else — identical to a filling
    /// `PreFilling` hydro, proving the reformulation is reused, not reinvented.
    #[test]
    fn dormant_nonfilling_frozen_identity_and_inflow_routed_downstream() {
        let (csc, row_lower, row_upper, off) =
            build_nonfilling_shortcircuit_case(0, RET_ENTRY_STAGE_ID, 9.0);
        let row_h = off.water_row_h2;
        let row_d = off.water_row_h3;

        // Frozen identity on H2's own row.
        assert_eq!(
            csc_at(&csc, off.h2_idx, row_h),
            1.0,
            "v_{{H2}} +1.0 on the frozen-identity row (dense storage column)"
        );
        assert_eq!(
            csc_at(&csc, off.col_storage_in_h2, row_h),
            -1.0,
            "v_{{H2,in}} −1.0 on the frozen-identity row"
        );
        assert_eq!(row_lower[row_h], 0.0, "frozen RHS lower == 0");
        assert_eq!(row_upper[row_h], 0.0, "frozen RHS upper == 0");

        // z_{H2} routed onto H3's balance row at −ζ, and absent from H2's frozen row.
        assert_eq!(
            csc_at(&csc, off.z_h2, row_d),
            -off.zeta,
            "z_{{H2}} routed onto H3's water row at −ζ (river flows past the un-built dam)"
        );
        assert_eq!(
            csc_at(&csc, off.z_h2, row_h),
            0.0,
            "z_{{H2}} absent from H2's frozen-identity row (not trapped)"
        );
    }

    /// The cascade tail (sink) case: a dormant non-filling hydro with no downstream
    /// drops its water from the system and keeps its frozen identity — feasible, no
    /// trapped inflow.
    #[test]
    fn dormant_nonfilling_sink_drops_water_and_keeps_frozen_identity() {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro_nonfilling(1, Some(2), None),
                ret_hydro_nonfilling(2, None, Some(RET_ENTRY_STAGE_ID)),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = 13.0;
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(0, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let row_h = layout.rows.water_balance.start + h2_idx;
        let z_h2 = layout.col_z_inflow_start() + h2_idx;

        assert_eq!(csc_at(&csc, h2_idx, row_h), 1.0, "v_{{H2}} +1.0");
        assert_eq!(
            csc_at(&csc, layout.col_storage_in_start() + h2_idx, row_h),
            -1.0,
            "v_{{H2,in}} −1.0"
        );
        assert_eq!(row_lower[row_h], 0.0, "frozen RHS 0");
        assert_eq!(row_upper[row_h], 0.0, "frozen RHS 0");
        for h in 0..layout.n_h {
            let r = layout.rows.water_balance.start + h;
            assert_eq!(
                csc_at(&csc, z_h2, r),
                0.0,
                "z_{{H2}} on no water row in the sink case (row {r})"
            );
        }
    }

    /// From `entry` onward the same non-filling hydro is `Operating`: its water row
    /// regains the standard form (own +τ releases, no short-circuit), proving the
    /// window opens at `entry` with NO intervening `Filling` phase.
    #[test]
    fn dormant_nonfilling_operating_from_entry_has_standard_row() {
        // entry == 0 ⇒ H2 is Operating at stage 0 (commissioned immediately).
        let (csc, row_lower, row_upper, off) = build_nonfilling_shortcircuit_case(0, 0, 0.0);
        let row_h = off.water_row_h2;
        let row_d = off.water_row_h3;

        for blk in 0..off.n_blks {
            let tau_h = [300.0_f64, 444.0][blk] * M3S_TO_HM3;
            assert_eq!(
                csc_at(&csc, off.h2_turbine[blk], row_h),
                tau_h,
                "blk {blk}: Operating H2 carries +τ on its own row (standard fill)"
            );
        }
        // No short-circuit: z_{H2} is NOT routed onto H3's row.
        assert_eq!(
            csc_at(&csc, off.z_h2, row_d),
            0.0,
            "z_{{H2}} not routed downstream when Operating"
        );
        // Standard equality row, not the frozen identity (lower == upper, base 0).
        assert_eq!(row_lower[row_h], row_upper[row_h], "H2 row is an equality");
    }

    /// Resolved offsets for a CHAINED short-circuit probe `H1 → H2 → H3` where H1
    /// AND H2 are BOTH `PreFilling` at the probe stage and H3 is operating.
    struct ChainOffsets {
        zeta: f64,
        h1_idx: usize,
        h2_idx: usize,
        water_row_h1: usize,
        water_row_h2: usize,
        water_row_h3: usize,
        col_storage_in_h1: usize,
        col_storage_in_h2: usize,
        z_h1: usize,
        z_h2: usize,
    }

    /// Build the cascade `H1(id 1, filling) → H2(id 2, filling) → H3(id 3,
    /// operating)` at `stage_id`, with H1's and H2's resolved per-stage withdrawals
    /// set to `withdrawal_h1` / `withdrawal_h2`. At a `PreFilling` `stage_id` both H1
    /// and H2 are `PreFilling` (frozen rows), so H1 must cascade THROUGH H2 to H3.
    /// Returns the CSC, `(row_lower, row_upper)`, and the chained offsets.
    #[allow(clippy::type_complexity)]
    fn build_chained_shortcircuit_case(
        stage_id: i32,
        withdrawal_h1: f64,
        withdrawal_h2: f64,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        ChainOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(2, Some(3), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(3, None, None, false),
            ],
            Vec::new(),
        );
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .bounds
            .hydro_bounds_mut(h1_idx, 0)
            .water_withdrawal_m3s = withdrawal_h1;
        fixtures
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = withdrawal_h2;
        let ctx = fixtures.make_ctx();
        let stage_index = usize::try_from(stage_id).expect("non-negative");
        let stage = two_block_stage(stage_index, [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let h3_idx = fixtures.hydro_pos[&EntityId(3)];
        let offsets = ChainOffsets {
            zeta: layout.zeta,
            h1_idx,
            h2_idx,
            water_row_h1: layout.rows.water_balance.start + h1_idx,
            water_row_h2: layout.rows.water_balance.start + h2_idx,
            water_row_h3: layout.rows.water_balance.start + h3_idx,
            col_storage_in_h1: layout.col_storage_in_start() + h1_idx,
            col_storage_in_h2: layout.col_storage_in_start() + h2_idx,
            z_h1: layout.col_z_inflow_start() + h1_idx,
            z_h2: layout.col_z_inflow_start() + h2_idx,
        };
        (csc, row_lower, row_upper, offsets)
    }

    /// CHAINED short-circuit: cascade `H1 → H2 → H3` with H1 AND H2 BOTH `PreFilling`
    /// at the same stage and H3 operating. Both H1's water (z_{H1}, withdrawal_{H1})
    /// and H2's water (z_{H2}, withdrawal_{H2}) land on H3's row (the first
    /// non-`PreFilling` downstream) — H1 cascades THROUGH the absent H2, never onto
    /// H2's frozen row.
    #[test]
    fn chained_prefilling_routes_both_links_to_first_operating_downstream() {
        let (w_h1, w_h2) = (11.0_f64, 23.0_f64);
        let (csc, _rl, row_upper, off) =
            build_chained_shortcircuit_case(RET_PREFILLING_ID, w_h1, w_h2);

        // (1) Incremental inflow of BOTH links lands on H3's row (−ζ each), and on
        // NO frozen row. H1 routing THROUGH H2 is the chained behaviour: routing to
        // the immediate downstream H2 would corrupt H2's frozen identity.
        assert_eq!(
            csc_at(&csc, off.z_h1, off.water_row_h3),
            -off.zeta,
            "z_{{H1}} carries −ζ on H3's row (H1 cascades THROUGH the PreFilling H2)"
        );
        assert_eq!(
            csc_at(&csc, off.z_h2, off.water_row_h3),
            -off.zeta,
            "z_{{H2}} carries −ζ on H3's row (first non-PreFilling downstream)"
        );
        // Neither z column touches H2's frozen-identity row.
        assert_eq!(
            csc_at(&csc, off.z_h1, off.water_row_h2),
            0.0,
            "z_{{H1}} must NOT land on H2's frozen row (the corruption this guards against)"
        );
        assert_eq!(
            csc_at(&csc, off.z_h2, off.water_row_h2),
            0.0,
            "z_{{H2}} must NOT land on H2's own frozen row"
        );
        // Nor H1's frozen row.
        assert_eq!(
            csc_at(&csc, off.z_h1, off.water_row_h1),
            0.0,
            "z_{{H1}} must NOT land on H1's own frozen row"
        );

        // (2) Withdrawal demand of BOTH links lands on H3's RHS: H3's row_upper
        // drops by ζ·withdrawal_{H1} + ζ·withdrawal_{H2} versus a zero-withdrawal
        // build. Both transfers resolve to H3 (the same short-circuit target),
        // NEVER onto H2's frozen RHS. The expected delta mirrors the code's two
        // independent per-link `-=` subtractions (rows.rs), NOT a single
        // ζ·(w_h1 + w_h2) product, so the FP rounding matches bit-for-bit.
        let (_csc0, _rl0, row_upper0, off0) =
            build_chained_shortcircuit_case(RET_PREFILLING_ID, 0.0, 0.0);
        let delta = off.zeta * w_h1 + off.zeta * w_h2;
        assert_eq!(
            row_upper0[off0.water_row_h3] - row_upper[off.water_row_h3],
            delta,
            "H3's RHS drops by ζ·withdrawal_{{H1}} + ζ·withdrawal_{{H2}}"
        );
    }

    /// CHAINED short-circuit, clean frozen identities: with H1 AND H2 BOTH
    /// `PreFilling`, EACH of H1's and H2's own water rows is exactly the frozen
    /// identity (`+1 v_h`, `−1 v_h_in`, RHS 0) and exactly 2 matrix entries — NOT
    /// corrupted by the other link's routed inflow/upstream/withdrawal terms.
    #[test]
    fn chained_prefilling_keeps_both_frozen_rows_clean() {
        let (csc, row_lower, row_upper, off) =
            build_chained_shortcircuit_case(RET_PREFILLING_ID, 7.0, 13.0);

        // Helper: count the matrix entries on a given water row and assert it is the
        // clean frozen identity (exactly +1 on v_h, −1 on v_h_in, nothing else).
        let assert_clean_frozen = |row: usize, col_v: usize, col_v_in: usize, label: &str| {
            assert_eq!(
                csc_at(&csc, col_v, row),
                1.0,
                "{label}: outgoing storage v carries +1.0 on the frozen row"
            );
            assert_eq!(
                csc_at(&csc, col_v_in, row),
                -1.0,
                "{label}: incoming storage v_in carries −1.0 on the frozen row"
            );
            // Exactly 2 entries on the whole row: +1 v, −1 v_in, nothing else.
            let n_entries: usize = (0..csc.0.len().saturating_sub(1))
                .map(|col| {
                    let start = usize::try_from(csc.0[col]).unwrap();
                    let end = usize::try_from(csc.0[col + 1]).unwrap();
                    csc.1[start..end]
                        .iter()
                        .filter(|&&r| usize::try_from(r).is_ok_and(|r| r == row))
                        .count()
                })
                .sum();
            assert_eq!(
                n_entries, 2,
                "{label}: frozen-identity row must have EXACTLY 2 entries \
                 (+1 v, −1 v_in) — found {n_entries}, indicating corruption by a \
                 chained link's routed terms"
            );
            assert_eq!(row_lower[row], 0.0, "{label}: frozen RHS lower == 0");
            assert_eq!(row_upper[row], 0.0, "{label}: frozen RHS upper == 0");
        };

        // H2's row clean (NOT corrupted by H1's terms — the bug this fix removes).
        assert_clean_frozen(off.water_row_h2, off.h2_idx, off.col_storage_in_h2, "H2");
        // H1's row likewise clean.
        assert_clean_frozen(off.water_row_h1, off.h1_idx, off.col_storage_in_h1, "H1");
    }

    /// CHAIN-to-sink: a cascade `H1 → H2` where H1 AND H2 are BOTH `PreFilling` and
    /// H2 is terminal (no downstream). The whole chain is `PreFilling`, so the
    /// resolved target is None (SINK): H1's and H2's water exits the system, and
    /// NEITHER frozen row is corrupted.
    #[test]
    fn chained_prefilling_all_the_way_down_is_sink_and_keeps_frozen_rows_clean() {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(2, None, Some(RET_ENTRY_STAGE_ID), true),
            ],
            Vec::new(),
        );
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .bounds
            .hydro_bounds_mut(h1_idx, 0)
            .water_withdrawal_m3s = 8.0;
        fixtures
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = 19.0;
        let ctx = fixtures.make_ctx();
        let stage = two_block_stage(usize::try_from(RET_PREFILLING_ID).unwrap(), [300.0, 444.0]);
        let state = state_layout_for(&ctx);
        let layout = StageLayout::new(&ctx, &state, &stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let z_h1 = layout.col_z_inflow_start() + h1_idx;
        let z_h2 = layout.col_z_inflow_start() + h2_idx;

        // Both links' inflow exits the system: neither z column lands on ANY water
        // row (no non-PreFilling downstream exists to receive it).
        for h in 0..layout.n_h {
            let r = layout.rows.water_balance.start + h;
            assert_eq!(
                csc_at(&csc, z_h1, r),
                0.0,
                "z_{{H1}} must not land on any water row in the all-PreFilling sink (row {r})"
            );
            assert_eq!(
                csc_at(&csc, z_h2, r),
                0.0,
                "z_{{H2}} must not land on any water row in the all-PreFilling sink (row {r})"
            );
        }

        // Both frozen rows are clean (exactly +1 v, −1 v_in, RHS 0) and no
        // withdrawal demand was folded onto any frozen RHS (the sink transfers
        // nothing).
        for (h_idx, label) in [(h1_idx, "H1"), (h2_idx, "H2")] {
            let row = layout.rows.water_balance.start + h_idx;
            assert_eq!(csc_at(&csc, h_idx, row), 1.0, "{label}: v +1.0");
            assert_eq!(
                csc_at(&csc, layout.col_storage_in_start() + h_idx, row),
                -1.0,
                "{label}: v_in −1.0"
            );
            assert_eq!(
                row_lower[row], 0.0,
                "{label}: frozen RHS 0 (no withdrawal folded in)"
            );
            assert_eq!(row_upper[row], 0.0, "{label}: frozen RHS 0");
        }
    }

    // ── Chronological PreFilling per-block frozen identity + short-circuit ────────

    /// Resolved offsets for a chronological H1→H2→H3 short-circuit probe (`n_blks = 2`).
    struct ChrScOffsets {
        n_blks: usize,
        h2_idx: usize,
        d_idx: usize,
        z_h2: usize,
        col_storage_in_h2: usize,
        h1_turbine: Vec<usize>,
        h1_spillage: Vec<usize>,
    }

    /// Build the cascade `H1(id 1) → H2(id 2, filling) → H3(id 3)` at the canonical
    /// `PreFilling` stage under `BlockMode::Chronological`, with H2's resolved
    /// per-stage withdrawal set to `withdrawal_h`. Returns the assembled CSC, the
    /// `(row_lower, row_upper)` vectors, the resolved [`StageLayout`] (block-major
    /// addressing reads through its accessors), and the offsets the assertions read.
    #[allow(clippy::type_complexity)]
    fn build_chronological_shortcircuit_case(
        withdrawal_h: f64,
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        StageLayout<'static>,
        ChrScOffsets,
    ) {
        let mut fixtures = PumpFixtures::new(
            vec![
                ret_hydro(1, Some(2), None, false),
                ret_hydro(2, Some(3), Some(RET_ENTRY_STAGE_ID), true),
                ret_hydro(3, None, None, false),
            ],
            Vec::new(),
        );
        let h2_idx = fixtures.hydro_pos[&EntityId(2)];
        fixtures
            .bounds
            .hydro_bounds_mut(h2_idx, 0)
            .water_withdrawal_m3s = withdrawal_h;
        let ctx = Box::leak(Box::new(fixtures.make_ctx()));
        let mut stage =
            two_block_stage(usize::try_from(RET_PREFILLING_ID).unwrap(), [300.0, 444.0]);
        stage.block_mode = cobre_core::BlockMode::Chronological;
        let stage = Box::leak(Box::new(stage));
        let state = Box::leak(Box::new(state_layout_for(ctx)));
        let layout = StageLayout::new(ctx, state, stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(ctx, stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(ctx, stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        let h1_idx = fixtures.hydro_pos[&EntityId(1)];
        let d_idx = fixtures.hydro_pos[&EntityId(3)];
        let offsets = ChrScOffsets {
            n_blks: layout.n_blks,
            h2_idx,
            d_idx,
            z_h2: layout.col_z_inflow_start() + h2_idx,
            col_storage_in_h2: layout.col_storage_in_start() + h2_idx,
            h1_turbine: (0..layout.n_blks)
                .map(|blk| layout.turbine_col(HydroSys::new(h1_idx), blk))
                .collect(),
            h1_spillage: (0..layout.n_blks)
                .map(|blk| layout.spillage_col(HydroSys::new(h1_idx), blk))
                .collect(),
        };
        (csc, row_lower, row_upper, layout, offsets)
    }

    /// AC#1: a chronological `K = 2` `PreFilling` hydro H2 emits `K` block-major
    /// frozen-identity rows `row_water + h·K + (k−1)`, each carrying EXACTLY two
    /// entries: `+1.0` on `Sᵏ` and `−1.0` on `Sᵏ⁻¹`, and nothing else (no
    /// flow/inflow/loss/withdrawal term — any coupling makes `β_h` stale-nonzero).
    /// The per-hydro `h·K` stride is what eliminates the single-row collision into a
    /// neighbour's block row.
    #[test]
    fn chronological_prefilling_frozen_identity_per_block() {
        let (csc, row_lower, row_upper, layout, off) = build_chronological_shortcircuit_case(9.0);
        let h = off.h2_idx;
        let n_blks = off.n_blks;

        for k in 1..=n_blks {
            let blk = k - 1;
            let row = layout.rows.water_balance.start + h * n_blks + blk;
            assert_eq!(
                csc_at(&csc, layout.block_storage_col(HydroSys::new(h), k), row),
                1.0,
                "block {k}: Sᵏ carries +1.0 on the frozen-identity row"
            );
            assert_eq!(
                csc_at(&csc, layout.block_storage_col(HydroSys::new(h), k - 1), row),
                -1.0,
                "block {k}: Sᵏ⁻¹ carries −1.0 on the frozen-identity row"
            );
            // No other column touches this block row (frozen identity Sᵏ − Sᵏ⁻¹ = 0):
            // count CSC entries landing on it across EVERY column.
            let entries_on_row = (0..csc.0.len() - 1)
                .filter(|&col| csc_at(&csc, col, row) != 0.0)
                .count();
            assert_eq!(
                entries_on_row, 2,
                "block {k}: frozen-identity row must have EXACTLY two entries (Sᵏ, Sᵏ⁻¹)"
            );
            assert_eq!(row_lower[row], 0.0, "block {k}: frozen RHS lower == 0");
            assert_eq!(row_upper[row], 0.0, "block {k}: frozen RHS upper == 0");
        }
    }

    /// AC#2: H2's per-block short-circuit lands `−τ_k` on the DOWNSTREAM target H3's
    /// block-`k` rows — H2's incremental inflow (`z_{H2}`) and H2's upstream H1
    /// releases — and NOTHING on H2's own (frozen) block rows. H1 is NOT a standard
    /// upstream of H3 (the cascade edge is H1→H2→H3), so a nonzero `−τ_k` on H1's
    /// columns at H3's block row can only come from the re-route.
    #[test]
    fn chronological_prefilling_shortcircuit_per_block() {
        let withdrawal_h = 17.5_f64;
        let (csc, _rl, row_upper, layout, off) =
            build_chronological_shortcircuit_case(withdrawal_h);
        let n_blks = off.n_blks;

        for k in 1..=n_blks {
            let blk = k - 1;
            let tau_k = [300.0_f64, 444.0][blk] * M3S_TO_HM3;
            let row_d = layout.rows.water_balance.start + off.d_idx * n_blks + blk;
            let row_h = layout.rows.water_balance.start + off.h2_idx * n_blks + blk;

            assert_eq!(
                csc_at(&csc, off.z_h2, row_d),
                -tau_k,
                "block {k}: z_{{H2}} carries −τ_k on H3's block row (re-routed inflow)"
            );
            assert_eq!(
                csc_at(&csc, off.z_h2, row_h),
                0.0,
                "block {k}: z_{{H2}} carries nothing on H2's own frozen block row"
            );
            assert_eq!(
                csc_at(&csc, off.h1_turbine[blk], row_d),
                -tau_k,
                "block {k}: H1 turbine carries −τ_k on H3's block row (cascade re-route)"
            );
            assert_eq!(
                csc_at(&csc, off.h1_spillage[blk], row_d),
                -tau_k,
                "block {k}: H1 spillage carries −τ_k on H3's block row"
            );
            assert_eq!(
                csc_at(&csc, off.h1_turbine[blk], row_h),
                0.0,
                "block {k}: H1 turbine absent from H2's own frozen block row"
            );
            assert_eq!(
                csc_at(&csc, off.col_storage_in_h2, row_d),
                0.0,
                "block {k}: H2's incoming storage stays on H2's row, not routed to H3"
            );
        }

        // Withdrawal demand transfers per block: H3's block-`k` row_upper drops by
        // τ_k·withdrawal_h versus a no-withdrawal build (H3's own withdrawal is 0).
        let (_csc0, _rl0, row_upper0, layout0, off0) = build_chronological_shortcircuit_case(0.0);
        for k in 1..=n_blks {
            let blk = k - 1;
            let tau_k = [300.0_f64, 444.0][blk] * M3S_TO_HM3;
            let row_d = layout.rows.water_balance.start + off.d_idx * n_blks + blk;
            let row_d0 = layout0.rows.water_balance.start + off0.d_idx * n_blks + blk;
            assert_eq!(
                row_upper0[row_d0] - row_upper[row_d],
                tau_k * withdrawal_h,
                "block {k}: H3's block RHS drops by τ_k·withdrawal_{{H2}}"
            );
        }
    }

    /// AC#3: a chronological `K = 1` `PreFilling` build is byte-identical to the
    /// parallel `PreFilling` build — `τ_1 = ζ`, no interior boundary, the single
    /// chained frozen row IS the parallel frozen row, and the per-block short-circuit
    /// collapses to the single-row `−ζ`/`−ζ·withdrawal` parallel form. Covers the CSC
    /// arrays and both RHS vectors.
    #[test]
    fn chronological_k1_prefilling_byte_identical() {
        let withdrawal_h = 11.0_f64;
        let build = |block_mode: cobre_core::BlockMode| {
            let mut fixtures = PumpFixtures::new(
                vec![
                    ret_hydro(1, Some(2), None, false),
                    ret_hydro(2, Some(3), Some(RET_ENTRY_STAGE_ID), true),
                    ret_hydro(3, None, None, false),
                ],
                Vec::new(),
            );
            let h2_idx = fixtures.hydro_pos[&EntityId(2)];
            fixtures
                .bounds
                .hydro_bounds_mut(h2_idx, 0)
                .water_withdrawal_m3s = withdrawal_h;
            let ctx = fixtures.make_ctx();
            let mut stage =
                two_block_stage(usize::try_from(RET_PREFILLING_ID).unwrap(), [372.0, 372.0]);
            stage.blocks.truncate(1);
            stage.block_mode = block_mode;
            let state = state_layout_for(&ctx);
            let layout = StageLayout::new(&ctx, &state, &stage, 0);
            let (rl, ru) = super::super::rows::fill_stage_rows(&ctx, &stage, 0, &layout);
            let csc = {
                let mut entries = build_stage_matrix_entries(&ctx, &stage, 0, &layout);
                for col in &mut entries {
                    col.sort_unstable_by_key(|&(row, _)| row);
                }
                assemble_csc(&entries)
            };
            (csc, rl, ru)
        };

        let (csc_p, rl_p, ru_p) = build(cobre_core::BlockMode::Parallel);
        let (csc_c, rl_c, ru_c) = build(cobre_core::BlockMode::Chronological);

        assert_eq!(
            csc_p.0, csc_c.0,
            "K=1 PreFilling col_starts must be byte-identical"
        );
        assert_eq!(
            csc_p.1, csc_c.1,
            "K=1 PreFilling row_indices must be byte-identical"
        );
        assert_eq!(
            csc_p.2, csc_c.2,
            "K=1 PreFilling values must be byte-identical"
        );
        assert_eq!(
            rl_p, rl_c,
            "K=1 PreFilling row_lower must be byte-identical"
        );
        assert_eq!(
            ru_p, ru_c,
            "K=1 PreFilling row_upper must be byte-identical"
        );
    }

    // ── Per-block FPHA & evaporation (block-local average storage) ───────────────

    use crate::hydro_models::{FphaPlane, LinearizedEvaporation, ResolvedProductionModel};

    const FPHA_GAMMA_V: f64 = 0.2;
    const EVAP_SLOPE: f64 = 0.03;
    const EVAP_INTERCEPT: f64 = 1.5;

    /// One non-filling hydro with a single-plane FPHA production model and a
    /// linearized evaporation model, built under `block_mode` with two blocks
    /// `[300.0, 444.0]` (truncated to one for the `K = 1` cases). Returns the
    /// assembled CSC, the `(row_lower, row_upper)` vectors, the column
    /// `(col_lower, col_upper, objective)` vectors, and the resolved `StageLayout`.
    #[allow(clippy::type_complexity)]
    fn build_fpha_evap_case(
        block_mode: cobre_core::BlockMode,
        durations: &[f64],
    ) -> (
        (Vec<i32>, Vec<i32>, Vec<f64>),
        Vec<f64>,
        Vec<f64>,
        (Vec<f64>, Vec<f64>, Vec<f64>),
        StageLayout<'static>,
    ) {
        let mut fixtures = PumpFixtures::new(vec![ret_hydro(1, None, None, false)], Vec::new())
            .with_evap_penalties(7.0, 11.0);
        fixtures.production_models = ProductionModelSet::new(
            vec![vec![ResolvedProductionModel::Fpha {
                planes: vec![FphaPlane {
                    intercept: 1.0,
                    gamma_v: FPHA_GAMMA_V,
                    gamma_q: 0.5,
                    gamma_s: 0.05,
                }],
            }]],
            1,
            N_STAGES,
        );
        fixtures.evaporation_models =
            EvaporationModelSet::new(vec![EvaporationModel::Linearized {
                coefficients: vec![LinearizedEvaporation {
                    intercept_m3s: EVAP_INTERCEPT,
                    volume_slope_m3s_per_hm3: EVAP_SLOPE,
                }],
                reference_volumes_hm3: vec![0.0],
            }]);
        let ctx = Box::leak(Box::new(fixtures.make_ctx()));
        let mut stage = two_block_stage(0, [300.0, 444.0]);
        stage.blocks.truncate(durations.len());
        for (blk, &d) in durations.iter().enumerate() {
            stage.blocks[blk].duration_hours = d;
        }
        stage.block_mode = block_mode;
        let stage = Box::leak(Box::new(stage));
        let state = Box::leak(Box::new(state_layout_for(ctx)));
        let layout = StageLayout::new(ctx, state, stage, 0);
        let (row_lower, row_upper) = super::super::rows::fill_stage_rows(ctx, stage, 0, &layout);
        let cols = super::super::columns::fill_stage_columns(ctx, stage, 0, &layout);
        let csc = {
            let mut entries = build_stage_matrix_entries(ctx, stage, 0, &layout);
            for col in &mut entries {
                col.sort_unstable_by_key(|&(row, _)| row);
            }
            assemble_csc(&entries)
        };
        (csc, row_lower, row_upper, cols, layout)
    }

    /// AC#1: a chronological `K = 2` FPHA plane row for block `k` carries `−γᵥ/2`
    /// on BOTH `block_storage_col(h, k−1)` (Sᵏ⁻¹) and `block_storage_col(h, k)` (Sᵏ)
    /// — the block-local average storage, both columns (D06), never one.
    #[test]
    fn chronological_fpha_uses_block_local_average() {
        let (csc, _rl, _ru, _cols, layout) =
            build_fpha_evap_case(cobre_core::BlockMode::Chronological, &[300.0, 444.0]);
        let h = 0_usize;
        let half_gamma_v = -FPHA_GAMMA_V / 2.0;
        // One plane, so block `k`'s FPHA row is at row_fpha_start + blk.
        for k in 1..=2 {
            let blk = k - 1;
            let row = layout.row_fpha_start() + blk;
            assert_eq!(
                csc_at(&csc, layout.block_storage_col(HydroSys::new(h), k - 1), row),
                half_gamma_v,
                "block {k}: −γᵥ/2 on Sᵏ⁻¹"
            );
            assert_eq!(
                csc_at(&csc, layout.block_storage_col(HydroSys::new(h), k), row),
                half_gamma_v,
                "block {k}: −γᵥ/2 on Sᵏ"
            );
        }
        // S⁰, the interior boundary S¹, and Sᴷ are three distinct columns, so the
        // two block rows average genuinely block-local storage (no aliasing).
        let s0 = layout.block_storage_col(HydroSys::new(h), 0);
        let s1 = layout.block_storage_col(HydroSys::new(h), 1);
        let s2 = layout.block_storage_col(HydroSys::new(h), 2);
        assert_ne!(s0, s1, "S⁰ and S¹ distinct");
        assert_ne!(s1, s2, "S¹ and Sᴷ distinct");
        assert_ne!(s0, s2, "S⁰ and Sᴷ distinct");
    }

    /// AC#2: a chronological `K = 2` evaporating hydro emits `K` evaporation rows,
    /// each with `−slope/2` on its own `(Sᵏ⁻¹, Sᵏ)` pair; each block's evaporation
    /// flow appears in that block's water row with `+τ_k`; each block's flow column
    /// is BOUNDED `[−q_max, +q_max]` (the wrong-bounds bug: leaving the extra
    /// per-block flow columns unbounded); and each block's directional slack
    /// objective is the cost times THAT block's `duration_hours` (block-scoped like
    /// the flow's `+τ_k` term — not the stage-total hours on every block, which would
    /// inflate the penalty `K`-fold).
    #[test]
    fn chronological_evaporation_per_block() {
        let (csc, rl, ru, cols, layout) =
            build_fpha_evap_case(cobre_core::BlockMode::Chronological, &[300.0, 444.0]);
        let (col_lower, col_upper, objective) = cols;
        let h = 0_usize;
        let local = 0_usize;
        let n_blks = 2_usize;
        let half_slope = -EVAP_SLOPE / 2.0;
        let q_max = (EVAP_INTERCEPT + EVAP_SLOPE * 100.0).abs() * 2.0;

        for k in 1..=n_blks {
            let blk = k - 1;
            let evap_row = layout.row_evap_start() + local * n_blks + blk;
            let tau_k = [300.0_f64, 444.0][blk] * M3S_TO_HM3;

            // Block-local average storage on the evaporation row.
            assert_eq!(
                csc_at(
                    &csc,
                    layout.block_storage_col(HydroSys::new(h), k - 1),
                    evap_row
                ),
                half_slope,
                "block {k}: −slope/2 on Sᵏ⁻¹"
            );
            assert_eq!(
                csc_at(
                    &csc,
                    layout.block_storage_col(HydroSys::new(h), k),
                    evap_row
                ),
                half_slope,
                "block {k}: −slope/2 on Sᵏ"
            );
            // The equality-row intercept is replicated per block.
            assert_eq!(rl[evap_row], EVAP_INTERCEPT, "block {k}: evap RHS lower");
            assert_eq!(ru[evap_row], EVAP_INTERCEPT, "block {k}: evap RHS upper");

            let flow_col = layout.evap_flow_col(EvapLocal::new(local), blk);
            // Flow enters block k's water row with +τ_k.
            let water_row = layout.rows.water_balance.start + h * n_blks + blk;
            assert_eq!(
                csc_at(&csc, flow_col, water_row),
                tau_k,
                "block {k}: evap flow carries +τ_k on its water row"
            );
            // Flow column bounded [−q_max, +q_max], NOT the default [0, +∞).
            assert_eq!(col_lower[flow_col], -q_max, "block {k}: flow lower −q_max");
            assert_eq!(col_upper[flow_col], q_max, "block {k}: flow upper +q_max");
            // Directional slacks carry nonzero objective, weighted by THIS block's
            // hours (not the stage-total, which would inflate the penalty K-fold).
            let block_hours = [300.0_f64, 444.0][blk];
            assert_eq!(
                objective[layout.evap_f_plus_col(EvapLocal::new(local), blk)],
                7.0 * block_hours,
                "block {k}: f_plus objective"
            );
            assert_eq!(
                objective[layout.evap_f_minus_col(EvapLocal::new(local), blk)],
                11.0 * block_hours,
                "block {k}: f_minus objective"
            );
        }
    }

    /// AC#3: a chronological `K = 1` FPHA + evaporation build is byte-identical to
    /// the parallel build — one water row, one FPHA plane row on `(S⁰, Sᴷ)`, one
    /// evaporation row/triple, and the flow's single `+ζ` water term. Covers the
    /// CSC arrays, both RHS vectors, and the column bound/objective vectors.
    #[test]
    fn chronological_k1_fpha_evap_byte_identical() {
        let (csc_p, rl_p, ru_p, (cl_p, cu_p, obj_p), _lp) =
            build_fpha_evap_case(cobre_core::BlockMode::Parallel, &[372.0]);
        let (csc_c, rl_c, ru_c, (cl_c, cu_c, obj_c), _lc) =
            build_fpha_evap_case(cobre_core::BlockMode::Chronological, &[372.0]);

        assert_eq!(csc_p.0, csc_c.0, "K=1 col_starts byte-identical");
        assert_eq!(csc_p.1, csc_c.1, "K=1 row_indices byte-identical");
        assert_eq!(csc_p.2, csc_c.2, "K=1 values byte-identical");
        assert_eq!(rl_p, rl_c, "K=1 row_lower byte-identical");
        assert_eq!(ru_p, ru_c, "K=1 row_upper byte-identical");
        assert_eq!(cl_p, cl_c, "K=1 col_lower byte-identical");
        assert_eq!(cu_p, cu_c, "K=1 col_upper byte-identical");
        assert_eq!(obj_p, obj_c, "K=1 objective byte-identical");
    }

    /// A parallel two-block FPHA + evaporation build is byte-identical to the
    /// pre-change parallel build in structure: the single FPHA row uses the stage
    /// endpoints `(S⁰, Sᴷ)` and the single evaporation row/triple with the flow's
    /// `+ζ` water term (the parallel path is unchanged by the per-block work).
    #[test]
    fn parallel_fpha_evap_uses_stage_endpoints() {
        let (csc, rl, _ru, cols, layout) =
            build_fpha_evap_case(cobre_core::BlockMode::Parallel, &[300.0, 444.0]);
        let (col_lower, col_upper, _obj) = cols;
        let h = 0_usize;
        let local = 0_usize;
        let col_s_in = layout.col_storage_in_start() + h;
        let col_s_out = h;

        // Single FPHA row (one plane) on the stage endpoints, −γᵥ/2 on both.
        let fpha_row = layout.row_fpha_start();
        assert_eq!(csc_at(&csc, col_s_in, fpha_row), -FPHA_GAMMA_V / 2.0);
        assert_eq!(csc_at(&csc, col_s_out, fpha_row), -FPHA_GAMMA_V / 2.0);

        // Single evaporation row (block 0 slot) on the stage endpoints.
        let evap_row = layout.row_evap_start();
        assert_eq!(csc_at(&csc, col_s_in, evap_row), -EVAP_SLOPE / 2.0);
        assert_eq!(csc_at(&csc, col_s_out, evap_row), -EVAP_SLOPE / 2.0);
        assert_eq!(rl[evap_row], EVAP_INTERCEPT);

        // Flow enters the single water row with +ζ (Σ_k τ_k).
        let flow_col = layout.evap_flow_col(EvapLocal::new(local), 0);
        let zeta = (300.0_f64 + 444.0) * M3S_TO_HM3;
        assert_eq!(
            csc_at(&csc, flow_col, layout.rows.water_balance.start + h),
            zeta,
            "parallel evap flow carries +ζ on the single water row"
        );
        let q_max = (EVAP_INTERCEPT + EVAP_SLOPE * 100.0).abs() * 2.0;
        assert_eq!(col_lower[flow_col], -q_max);
        assert_eq!(col_upper[flow_col], q_max);
    }
}
