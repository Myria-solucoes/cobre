use cobre_core::{CoefficientRef, ConstraintSense, Stage};

use crate::generic_constraints::resolve_variable_ref;
use crate::hydro_models::{EvaporationModel, ResolvedProductionModel};

use super::M3S_TO_HM3;
use super::fpha_cursor::for_each_fpha_plane;
use super::layout::{StageLayout, TemplateBuildCtx};

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
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let grid = layout.indexer.block_grid();
    for local_idx in 0..ctx.n_anticipated {
        let row = layout.anticipated.row_anticipated_fishing_start + local_idx;
        let thermal_idx = ctx.anticipated_thermal_indices[local_idx];
        let mut block_hours_total: f64 = 0.0;
        for blk in 0..n_blks {
            let col_gen = grid.flat(layout.col_thermal_start(), thermal_idx, blk);
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
/// Row bounds are filled by `super::rows::fill_anticipated_state_out_def_rows`. The
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
        let col = layout.col_withdrawal_neg_start() + h_idx;
        let row = row_water + h_idx;
        col_entries[col].push((row, -zeta));
    }

    // Over-withdrawal slack (pos): removes additional water from the balance.
    // When the solver withdraws more than the target, the pos slack accounts for
    // the excess withdrawal at a penalty cost.
    for h_idx in 0..n_h {
        let col = layout.col_withdrawal_pos_start() + h_idx;
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
    let grid = layout.indexer.block_grid();
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
            let col = grid.flat(layout.col_pumping_start, p_idx, blk);
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
    let grid = layout.indexer.block_grid();
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
                    let col = layout.turbine_col(h_idx, blk);
                    col_entries[col].push((row, rho));
                }
            }
        }
    }

    for (t_idx, thermal) in ctx.thermals.iter().enumerate() {
        if let Some(&b_idx) = ctx.bus_pos.get(&thermal.bus_id) {
            for blk in 0..n_blks {
                let row = grid.flat(row_load, b_idx, blk);
                let col = grid.flat(layout.col_thermal_start(), t_idx, blk);
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

    // Pumping power: negative injection on the station's bus. Iterate the
    // canonical ID-sorted `pumping_stations` slice so `p_idx` matches the
    // column block; a bus id absent from `bus_pos` skips that station with no
    // entry (semantic validation is a separate concern).
    for (p_idx, station) in ctx.pumping_stations.iter().enumerate() {
        if let Some(&b_idx) = ctx.bus_pos.get(&station.bus_id) {
            for blk in 0..n_blks {
                let row = grid.flat(row_load, b_idx, blk);
                let col = grid.flat(layout.col_pumping_start, p_idx, blk);
                col_entries[col].push((row, -station.consumption_mw_per_m3s));
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
            let col_exc = grid.flat(layout.col_excess_start(), b_idx, blk);
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
/// where `gamma_0` is already encoded in the row upper bound set by `super::rows::fill_stage_rows`.
///
/// FPHA uses **average** storage `(V_in + V_out)/2`, so `-gamma_v/2` lands on
/// BOTH the outgoing-storage column (`v = h_idx`) and the incoming-storage column
/// (`v_in = col_storage_in_start + h_idx`). Putting `-gamma_v/2` on `v` (`V_out`)
/// alone compiles and passes single-plane single-hydro tests, but understates
/// generation by the `V_in` head term — the wrong-but-compiling alternative that
/// deterministic case D06 pins against.
///
/// Driven by [`for_each_fpha_plane`] so the matrix coefficients and the row
/// bounds set by [`super::rows::fill_fpha_rows`] share one row cursor.
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
/// `super::rows::fill_stage_rows`. When `v_in` is fixed to value `V`, the effective RHS
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

        let row = layout.row_evap_start() + local_idx;

        col_entries[col_evaporation_flow].push((row, 1.0));
        col_entries[col_v].push((row, -coeff.volume_slope_m3s_per_hm3 / 2.0));
        col_entries[col_v_in].push((row, -coeff.volume_slope_m3s_per_hm3 / 2.0));
        col_entries[col_f_plus].push((row, 1.0));
        col_entries[col_f_minus].push((row, -1.0));
    }
}

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
    let grid = layout.indexer.block_grid();
    for (ncs_local, &ncs_sys_idx) in layout.active_ncs_indices.iter().enumerate() {
        let ncs = &ctx.non_controllable_sources[ncs_sys_idx];
        let Some(&bus_idx) = ctx.bus_pos.get(&ncs.bus_id) else {
            // Unknown bus — should not happen with valid data, but defensive skip.
            continue;
        };
        for blk in 0..layout.n_blks {
            let col = grid.flat(layout.col_ncs_start, ncs_local, blk);
            let row = grid.flat(layout.row_load_balance_start(), bus_idx, blk);
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
    stage_idx: usize,
    layout: &StageLayout,
    col_entries: &mut [Vec<(usize, f64)>],
) {
    let n_blks = layout.n_blks;
    let grid = layout.indexer.block_grid();

    for (h_idx, fpha_local_entry) in layout.fpha_local_index.iter().enumerate() {
        // ── Min outflow (per block): q + s + d + sigma >= min_outflow_m3s ───
        for blk in 0..n_blks {
            let row = grid.flat(layout.row_min_outflow_start(), h_idx, blk);
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
            let row = grid.flat(layout.row_max_outflow_start(), h_idx, blk);
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
            let row = grid.flat(layout.row_min_turbine_start(), h_idx, blk);
            let col_q = layout.turbine_col(h_idx, blk);
            col_entries[col_q].push((row, 1.0));
            let col_slack = layout.turbine_below_col(h_idx, blk);
            col_entries[col_slack].push((row, 1.0));
        }

        // ── Min generation (per block): g + sigma >= min_generation_mw ──────
        if let Some(&local_fpha_idx) = fpha_local_entry.as_ref() {
            // FPHA: generation variable g_{h,blk} (already in MW).
            for blk in 0..n_blks {
                let row = grid.flat(layout.row_min_generation_start(), h_idx, blk);
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
                let row = grid.flat(layout.row_min_generation_start(), h_idx, blk);
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
    fill_operational_violation_entries(ctx, stage_idx, layout, &mut col_entries);
    fill_anticipated_fishing_entries(ctx, stage, layout, &mut col_entries);

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
