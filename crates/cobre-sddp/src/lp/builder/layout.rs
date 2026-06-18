use std::collections::HashMap;

use cobre_core::{
    Bus, CascadeTopology, ConstraintSense, EntityId, GenericConstraint, Hydro, Line, LoadModel,
    NonControllableSource, PumpingStation, ResolvedBounds, ResolvedExchangeFactors,
    ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors,
    ResolvedPenalties, Stage, Thermal,
};
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::hydro_models::{
    EvaporationModel, EvaporationModelSet, ProductionModelSet, ResolvedProductionModel,
};
use crate::indexer::StageIndexer;

use super::{GenericConstraintRowEntry, M3S_TO_HM3};

/// System-level context shared across all stages during template construction.
///
/// Bundles the references extracted from a `System` before the per-stage
/// loop begins. Constructed once in `build_stage_templates` and borrowed by
/// `build_single_stage_template` for each study stage.
pub(crate) struct TemplateBuildCtx<'a> {
    pub(crate) hydros: &'a [Hydro],
    pub(crate) thermals: &'a [Thermal],
    pub(crate) lines: &'a [Line],
    pub(crate) buses: &'a [Bus],
    pub(crate) load_models: &'a [LoadModel],
    pub(crate) cascade: &'a CascadeTopology,
    pub(crate) bounds: &'a ResolvedBounds,
    pub(crate) penalties: &'a ResolvedPenalties,
    pub(crate) hydro_pos: HashMap<EntityId, usize>,
    pub(crate) thermal_pos: HashMap<EntityId, usize>,
    pub(crate) line_pos: HashMap<EntityId, usize>,
    pub(crate) bus_pos: HashMap<EntityId, usize>,
    pub(crate) par_lp: &'a PrecomputedPar,
    /// Resolved production models for all (hydro, stage) pairs.
    pub(crate) production_models: &'a ProductionModelSet,
    /// Resolved evaporation models for all hydro plants.
    pub(crate) evaporation_models: &'a EvaporationModelSet,
    /// Generic constraint definitions (expression, sense, slack config).
    pub(crate) generic_constraints: &'a [GenericConstraint],
    /// Pre-resolved table mapping `(constraint_idx, stage_id)` to active bound entries.
    pub(crate) resolved_generic_bounds: &'a ResolvedGenericConstraintBounds,
    /// Pre-resolved per-block load scaling factors.
    pub(crate) resolved_load_factors: &'a ResolvedLoadFactors,
    /// Pre-resolved per-block exchange capacity factors.
    pub(crate) resolved_exchange_factors: &'a ResolvedExchangeFactors,
    /// Non-controllable source entities sorted by ID.
    pub(crate) non_controllable_sources: &'a [NonControllableSource],
    /// Pre-resolved per-stage NCS available generation bounds.
    pub(crate) resolved_ncs_bounds: &'a ResolvedNcsBounds,
    /// Pre-resolved per-block NCS generation scaling factors.
    pub(crate) resolved_ncs_factors: &'a ResolvedNcsFactors,
    /// Pumping station entities sorted by ID.
    ///
    /// Canonical-order slice from `System::pumping_stations`; iterating it in
    /// slot order (the per-station local index `p_idx`) upholds the
    /// declaration-order bit-determinism rule. The matrix-fill helpers
    /// (`fill_pumping_columns` for bounds, `fill_pumping_water_entries` for the
    /// source/destination water-balance ±τ coupling) iterate this slice; the
    /// `PumpingFlow`/`PumpingPower` resolver arm indexes into it via `pumping_pos`.
    pub(crate) pumping_stations: &'a [PumpingStation],
    /// Station id → local index into `pumping_stations`.
    ///
    /// Built from the ID-sorted `pumping_stations` slice (id → slot), exactly as
    /// `hydro_pos`/`bus_pos`. The local index is the per-station column-block
    /// position used by `col_pumping_start + pos * n_blks + blk`. The
    /// `PumpingFlow`/`PumpingPower` resolver arm indexes `pumping_stations` via
    /// this map (passed in `PumpingRefs`).
    pub(crate) pumping_pos: HashMap<EntityId, usize>,
    /// Number of pumping stations (`pumping_stations.len()`).
    ///
    /// The authoritative count threaded into `EquipmentCounts.n_pumping`, so the
    /// layout reserves `n_pumping * n_blks` pumping-flow columns. Asserted equal
    /// to `bounds.n_pumping()` at ctx construction — a divergence means the
    /// resolved-bounds table and the entity slice disagree on station count.
    pub(crate) n_pumping: usize,
    /// Lookup table for parameter coefficient resolution.
    ///
    /// Maps `(parameter_id, stage_idx)` to a pre-resolved `f64` value.
    /// Queried by the LP builder when a [`cobre_core::CoefficientRef::Parameter`]
    /// term is encountered in a generic constraint expression.
    pub(crate) resolved_parameters: &'a crate::resolved_parameters::ResolvedParameters,
    /// Mapping from target hydro ID to source hydro indices that divert to it.
    ///
    /// For each hydro `d` with `diversion.downstream_id == target_id`, the map
    /// contains `d`'s system-level hydro index in the vec for `target_id`.
    /// Built once in `build_stage_templates()`.
    pub(crate) diversion_upstream: HashMap<EntityId, Vec<usize>>,
    pub(crate) n_hydros: usize,
    pub(crate) n_thermals: usize,
    pub(crate) n_lines: usize,
    pub(crate) n_buses: usize,
    pub(crate) max_par_order: usize,
    /// Number of thermals with `anticipated_config.is_some()`.
    pub(crate) n_anticipated: usize,
    /// Maximum `lead_stages` across the anticipated thermals (`K_max`).
    pub(crate) k_max: usize,
    /// Per-plant `lead_stages` (`K_i`) for the anticipated thermals.
    ///
    /// Length `n_anticipated`. Entry `i` is the `lead_stages` for the
    /// `i`-th anticipated thermal (in declaration order within the anticipated subset).
    pub(crate) anticipated_lead_stages: Vec<usize>,
    /// Mapping from anticipated-local position to global thermal index.
    ///
    /// Length `n_anticipated`. Entry `i` is the position within `ctx.thermals`
    /// of the `i`-th anticipated plant. Mirrors the FPHA `fpha_hydro_indices` pattern.
    pub(crate) anticipated_thermal_indices: Vec<usize>,
    pub(crate) has_penalty: bool,
    /// Cumulative discount factor at each stage for NPV cost computation.
    ///
    /// `cumulative_discount_factors[t]` is the present-value multiplier at stage `t`.
    /// Length is `n_study_stages`: the strict anticipated-decision predicate
    /// (`stage_idx + K_i < n_stages`) guarantees every delivery lookup falls
    /// within `[0, n_stages)`.
    /// Populated by `build_template_build_ctx` before the per-stage template loop.
    pub(crate) cumulative_discount_factors: Vec<f64>,
    /// Total stage hours for each study stage.
    ///
    /// `total_hours_per_stage[stage_idx]` is the sum of `block.duration_hours` for all
    /// blocks in that stage. Length is `n_study_stages`: the strict anticipated-decision
    /// predicate (`stage_idx + K_i < n_stages`) guarantees every delivery lookup falls
    /// within `[0, n_stages)`.
    /// Populated by `build_template_build_ctx` before the per-stage template loop.
    pub(crate) total_hours_per_stage: Vec<f64>,
}

/// Pre-computed column and row layout offsets for a single stage LP.
///
/// Centralises the arithmetic that derives column-start and row-start indices
/// from entity counts and block count so that the filling helpers do not need
/// to recompute them independently.
pub(crate) struct StageLayout {
    pub(crate) n_blks: usize,
    pub(crate) n_h: usize,
    pub(crate) lag_order: usize,
    /// Number of anticipated thermals (mirrors `TemplateBuildCtx.n_anticipated`).
    ///
    /// Stored here so matrix helpers can read it from the layout without
    /// borrowing `ctx`. Consumed by `anticipated_decision` column allocation,
    /// fishing-row construction, and anticipated-state-fixing CSC helpers.
    pub(crate) n_anticipated: usize,
    /// Maximum `lead_stages` across the anticipated thermals (`K_max`).
    ///
    /// Consumed by the anticipated-state column bounds, state-fixing row CSC entries,
    /// and ring-buffer slot arithmetic.
    pub(crate) k_max: usize,
    /// Anticipated-state column count: `n_anticipated * k_max`.
    ///
    /// Equals the width of the `anticipated_state` ring-buffer block that is
    /// inserted between `inflow_lags` and `z_inflow` in the LP column layout.
    /// Zero when `n_anticipated == 0`. Exposed for test introspection; the
    /// helpers iterate via `k_max × n_anticipated` separately.
    // Rationale: asserted in layout unit tests to verify anticipated-state column shifts;
    // production matrix helpers derive the same count inline from `n_anticipated * k_max`
    // rather than reading this field, so the lint fires on the production side.
    #[allow(dead_code)]
    pub(crate) n_ant_state: usize,
    /// Column index of the theta (future-cost) variable.
    ///
    /// Derived from the augmented indexer: `idx.theta = N*(3+L) + n_ant_state`.
    /// Used by matrix helpers instead of reconstructing a base indexer locally.
    pub(crate) col_theta: usize,
    /// Column index of the first incoming-storage variable (`v_in_0`).
    ///
    /// Derived from the augmented indexer: `idx.storage_in.start`.
    /// Used by matrix helpers instead of reconstructing a base indexer locally.
    pub(crate) col_storage_in_start: usize,
    /// Column index of the first AR lag variable.
    ///
    /// Derived from the augmented indexer: `idx.inflow_lags.start`.
    /// Used by matrix helpers instead of reconstructing a base indexer locally.
    pub(crate) col_inflow_lags_start: usize,
    // Column regions
    pub(crate) col_turbine_start: usize,
    pub(crate) col_spillage_start: usize,
    /// Start of diversion flow columns (one per hydro per block).
    ///
    /// Layout within this region: `col_diversion_start + h_idx * n_blks + blk`.
    /// Hydros without diversion have bounds [0, 0]; presolve eliminates them.
    pub(crate) col_diversion_start: usize,
    pub(crate) col_thermal_start: usize,
    /// Start of anticipated-decision columns: one per anticipated thermal, stage-level.
    ///
    /// Layout: `col_anticipated_decision_start + local_anticipated_idx`.
    /// Equals `col_thermal_end = col_thermal_start + n_thermals * n_blks`.
    /// Zero anticipated thermals: equals `col_thermal_start` (degenerate but valid;
    /// the column range is empty and `col_line_fwd_start` is unshifted).
    pub(crate) col_anticipated_decision_start: usize,
    /// Start of the `anticipated_state_out` column block (one column per
    /// anticipated plant; stage-level, NOT per-block). Located immediately
    /// after `col_anticipated_decision_start` in the control region.
    /// Pinned to `decision_col[plant]` by the `anticipated_state_out_def` row.
    /// When `n_anticipated == 0`, equals `col_anticipated_decision_start`
    /// (the block is empty).
    pub(crate) col_anticipated_state_out_start: usize,
    /// Start of the `anticipated_state_out_def` equality row block.
    /// One row per ACTIVE plant (`stage_idx + K_p < n_stages`); inactive
    /// plants emit no row, matching the strict gate of
    /// `anticipated_decision_active_at_stage`. Located adjacent to and
    /// immediately after `row_anticipated_fishing_start`.
    pub(crate) row_anticipated_state_out_def_start: usize,
    /// Number of `anticipated_state_out_def` rows at this stage.
    ///
    /// Equals the count of plants with `stage_idx + K_p < n_stages`
    /// (strict gate). Zero when `n_anticipated == 0` or when no plant is
    /// active at this stage. Used by the matrix-fill helpers to drive the
    /// active-row iteration.
    // Rationale: read by `matrix.rs` debug_assert_eq! guards at three production call
    // sites; the dead_code lint fires here because the field is defined in this sibling
    // `layout` module and the lint analyser does not see cross-module field access.
    #[allow(dead_code)]
    pub(crate) n_anticipated_state_out_def_rows: usize,
    /// Start of anticipated-state columns (ring-buffer slots for committed MW).
    ///
    /// Slot-major layout: column for slot `s`, plant `i` is at
    /// `col_anticipated_state_start + s * n_anticipated + i`.
    /// Slot 0 (the currently-delivering commitment under the always-active
    /// fishing predicate) is at `col_anticipated_state_start + i`.
    /// Zero when `n_anticipated == 0` (empty column range).
    /// Derived from `idx.anticipated_state.start` in `StageLayout::new`.
    pub(crate) col_anticipated_state_start: usize,
    pub(crate) col_line_fwd_start: usize,
    pub(crate) col_line_rev_start: usize,
    pub(crate) col_deficit_start: usize,
    /// Maximum number of deficit segments across all buses (S).
    ///
    /// The deficit region spans `n_buses * max_deficit_segments * n_blks` columns.
    pub(crate) max_deficit_segments: usize,
    pub(crate) col_excess_start: usize,
    pub(crate) col_inflow_slack_start: usize,
    /// Start of FPHA generation columns (one per FPHA hydro per block).
    ///
    /// Layout within this region: `col_generation_start + local_fpha_idx * n_blks + blk`.
    pub(crate) col_generation_start: usize,
    // Row regions
    pub(crate) row_water_balance_start: usize,
    pub(crate) row_load_balance_start: usize,
    /// Start of FPHA constraint rows (after load-balance rows).
    ///
    /// Layout: `row_fpha_start + local_fpha_idx * n_blks * n_planes + blk * n_planes + plane_idx`.
    pub(crate) row_fpha_start: usize,
    /// Start of evaporation constraint rows (after FPHA rows).
    ///
    /// One equality row per evaporation hydro.
    /// Layout: `row_evap_start + local_evap_idx`.
    pub(crate) row_evap_start: usize,
    /// Start of evaporation columns (after FPHA generation columns).
    ///
    /// 3 stage-level columns per evaporation hydro (evaporation outflow, `f_evap_plus`, `f_evap_minus`).
    /// Layout: `col_evap_start + local_evap_idx * 3 + {0, 1, 2}`.
    pub(crate) col_evap_start: usize,
    /// Start of under-withdrawal slack columns (after evaporation columns).
    ///
    /// One stage-level column per operating hydro.
    /// Layout: `col_withdrawal_neg_start + h`.
    /// Zero when `n_h == 0`.
    pub(crate) col_withdrawal_neg_start: usize,
    /// Start of over-withdrawal slack columns (after under-withdrawal slacks).
    ///
    /// One stage-level column per operating hydro.
    /// Layout: `col_withdrawal_pos_start + h`.
    /// Zero when `n_h == 0`.
    pub(crate) col_withdrawal_pos_start: usize,
    /// Start of outflow-below-minimum slack columns (one per hydro per block).
    ///
    /// Inserted after withdrawal slack columns.
    /// Layout: `col_outflow_below_start + h_idx * n_blks + blk`.
    pub(crate) col_outflow_below_start: usize,
    /// Start of outflow-above-maximum slack columns (one per hydro per block).
    ///
    /// Layout: `col_outflow_above_start + h_idx * n_blks + blk`.
    pub(crate) col_outflow_above_start: usize,
    /// Start of turbine-below-minimum slack columns (one per hydro per block).
    ///
    /// Layout: `col_turbine_below_start + h_idx * n_blks + blk`.
    pub(crate) col_turbine_below_start: usize,
    /// Start of generation-below-minimum slack columns (one per hydro per block).
    ///
    /// Layout: `col_generation_below_start + h_idx * n_blks + blk`.
    pub(crate) col_generation_below_start: usize,
    /// Start of NCS generation columns (after operational violation slack columns).
    ///
    /// One column per active NCS per block.
    /// Layout: `col_ncs_start + ncs_local_idx * n_blks + blk`.
    pub(crate) col_ncs_start: usize,
    /// Number of active NCS entities at this stage.
    pub(crate) n_ncs: usize,
    /// Indices (into `ctx.non_controllable_sources`) of NCS active at this stage.
    pub(crate) active_ncs_indices: Vec<usize>,
    /// Start of pumping-flow columns (after the NCS region, before generic-slack columns).
    ///
    /// One column per pumping station per block, block-major:
    /// `col_pumping_start + station_local_idx * n_blks + blk`.
    /// When `n_pumping == 0` the block is empty and `col_pumping_start` equals
    /// `col_ncs_end`, leaving every downstream `col_*_start` and `num_cols`
    /// byte-identical to a station-free system.
    ///
    /// Read by `fill_pumping_columns` (column bounds) and
    /// `fill_pumping_water_entries` (water-balance ±τ coupling), which address the
    /// per-station per-block column as `col_pumping_start + p_idx * n_blks + blk`.
    pub(crate) col_pumping_start: usize,
    /// Number of pumping stations contributing columns at this stage.
    ///
    /// Each station contributes `n_blks` columns. Sourced from `ctx.n_pumping`.
    /// The column count is the full station count at every stage: pumping
    /// `entry_stage_id`/`exit_stage_id` are parsed but not applied to the LP, so
    /// no commissioning gating shrinks or zeroes this block per stage (that gating
    /// is unimplemented). Read by `build_single_stage_template` to populate
    /// `StageTemplates::n_pumping_per_stage`, which the simulation extraction
    /// pipeline uses to bound the per-(station, block) primal read.
    pub(crate) n_pumping: usize,
    pub(crate) num_cols: usize,
    /// Start of minimum-outflow constraint rows (one per hydro per block, after evaporation rows).
    ///
    /// Layout: `row_min_outflow_start + h_idx * n_blks + blk`.
    pub(crate) row_min_outflow_start: usize,
    /// Start of maximum-outflow constraint rows (one per hydro per block).
    ///
    /// Layout: `row_max_outflow_start + h_idx * n_blks + blk`.
    pub(crate) row_max_outflow_start: usize,
    /// Start of minimum-turbine constraint rows (one per hydro per block).
    ///
    /// Layout: `row_min_turbine_start + h_idx * n_blks + blk`.
    pub(crate) row_min_turbine_start: usize,
    /// Start of minimum-generation constraint rows (one per hydro per block).
    ///
    /// Layout: `row_min_generation_start + h_idx * n_blks + blk`.
    pub(crate) row_min_generation_start: usize,
    /// Start of anticipated-state-fixing equality rows.
    ///
    /// One equality row per (slot, plant) pair in `[0, k_max) × [0, n_anticipated)`.
    /// Slot-major layout: row for slot `s`, plant `i` is at
    /// `row_anticipated_state_fixing_start + s * n_anticipated + i`.
    /// Equals `idx.anticipated_state_fixing.start` from the augmented indexer
    /// (which mirrors `anticipated_state.start = N*(1+L)` numerically).
    /// Zero when `n_anticipated == 0` (empty row range).
    /// Row bounds are placeholder `0 == 0`; the RHS is patched during setup.
    // Rationale: asserted in layout unit tests to confirm the sentinel is 0 (state pinning
    // uses column bounds, not rows); production code never reads this field because the
    // `anticipated_state_fixing` row range is a permanent empty sentinel (`0..0`).
    #[allow(dead_code)]
    pub(crate) row_anticipated_state_fixing_start: usize,
    /// Start of anticipated-fishing constraint rows (after operational violation rows).
    ///
    /// One equality row per anticipated plant (always-active predicate).
    /// Layout: `row_anticipated_fishing_start + local_idx`.
    pub(crate) row_anticipated_fishing_start: usize,
    /// Number of anticipated-fishing rows at this stage.
    ///
    /// Always equals `n_anticipated` under the always-active rule.
    /// Zero when `n_anticipated == 0`.
    pub(crate) n_anticipated_fishing_rows: usize,
    /// Start of generic constraint rows (after operational violation rows).
    ///
    /// One row per active `(constraint, block)` pair.
    /// Equals `num_rows_before_generic` when no generic constraints are active.
    pub(crate) row_generic_start: usize,
    pub(crate) num_rows: usize,
    /// Total number of generic constraint rows for this stage.
    ///
    /// Zero when no generic constraints are active.
    pub(crate) n_generic_rows: usize,
    /// Start of z-inflow definition rows (after generic constraint rows).
    ///
    /// One equality row per hydro, defining `z_h = base_h + sigma_h * eta_h + sum_l[psi_l * lag_in[h,l]]`.
    pub(crate) row_z_inflow_start: usize,
    /// Start of z-inflow columns (after generic constraint slack columns).
    ///
    /// One free column per hydro (`z_h`, lower = -inf, upper = +inf, cost = 0.0).
    pub(crate) col_z_inflow_start: usize,
    // Template metadata
    pub(crate) n_state: usize,
    pub(crate) n_dual_relevant: usize,
    // Scalar derived quantities used by row-bound and matrix helpers
    pub(crate) zeta: f64,
    // FPHA hydro information for this stage
    /// Indices (into `ctx.hydros`) of hydros using FPHA at this stage.
    pub(crate) fpha_hydro_indices: Vec<usize>,
    /// Number of hyperplane planes per FPHA hydro at this stage.
    pub(crate) fpha_planes_per_hydro: Vec<usize>,
    // Evaporation hydro information for this stage
    /// Indices (into `ctx.hydros`) of hydros with linearized evaporation at this stage.
    pub(crate) evap_hydro_indices: Vec<usize>,
    /// Per-row metadata for active generic constraint rows at this stage.
    ///
    /// One entry per active `(constraint, block)` pair, in constraint-index-major
    /// order within each constraint's bound entries. Used for CSC matrix construction,
    /// row bound filling, and objective coefficient filling.
    pub(crate) generic_constraint_rows: Vec<GenericConstraintRowEntry>,
    /// Full augmented indexer for this stage.
    ///
    /// Cached here so that `fill_generic_constraint_entries` can call
    /// `resolve_variable_ref` without rebuilding the indexer (and cloning the
    /// anticipated metadata vecs) on every template build call.
    pub(crate) indexer: StageIndexer,
}

// ── Private helper return structs ─────────────────────────────────────────────

/// Layout metadata for all active generic constraint rows and slack columns.
struct GenericConstraintLayout {
    n_generic_rows: usize,
    n_generic_slack_cols: usize,
    generic_constraint_rows: Vec<GenericConstraintRowEntry>,
}

// ── Private helper functions ───────────────────────────────────────────────────

/// Collect the FPHA hydro indices and per-hydro plane counts for this stage.
///
/// The returned vectors feed the indexer's [`FphaColumnLayout`], which is the
/// single owner of the FPHA column and row offsets; this helper only enumerates
/// which hydros use FPHA, never their offsets.
///
/// [`FphaColumnLayout`]: crate::indexer::FphaColumnLayout
fn identify_fpha_hydros(ctx: &TemplateBuildCtx<'_>, stage_idx: usize) -> (Vec<usize>, Vec<usize>) {
    let mut fpha_hydro_indices: Vec<usize> = Vec::new();
    let mut fpha_planes_per_hydro: Vec<usize> = Vec::new();
    for h_idx in 0..ctx.n_hydros {
        if let ResolvedProductionModel::Fpha { planes, .. } =
            ctx.production_models.model(h_idx, stage_idx)
        {
            fpha_hydro_indices.push(h_idx);
            fpha_planes_per_hydro.push(planes.len());
        }
    }
    (fpha_hydro_indices, fpha_planes_per_hydro)
}

/// Collect the indices of hydros with linearized evaporation at this stage.
///
/// The returned vector feeds the indexer's [`EvapConfig`], which is the single
/// owner of the evaporation column and row offsets; this helper only enumerates
/// which hydros use evaporation, never their offsets.
///
/// [`EvapConfig`]: crate::indexer::EvapConfig
fn identify_evap_hydros(ctx: &TemplateBuildCtx<'_>) -> Vec<usize> {
    (0..ctx.n_hydros)
        .filter(|&h_idx| {
            matches!(
                ctx.evaporation_models.model(h_idx),
                EvaporationModel::Linearized { .. }
            )
        })
        .collect()
}

/// Collect indices of NCS entities that are active at this stage.
///
/// An NCS is active when the stage is at or after the entry stage (if any) and
/// strictly before the exit stage (if any).
fn identify_active_ncs(ctx: &TemplateBuildCtx<'_>, stage: &Stage) -> Vec<usize> {
    ctx.non_controllable_sources
        .iter()
        .enumerate()
        .filter_map(|(i, ncs)| {
            let ok = ncs.entry_stage_id.is_none_or(|e| e <= stage.id)
                && ncs.exit_stage_id.is_none_or(|e| stage.id < e);
            ok.then_some(i)
        })
        .collect()
}

/// Allocate the slack column index/indices for one generic-constraint row.
///
/// Returns `(slack_plus_col, slack_minus_col)`, advancing `n_slack_cols` by the
/// number of columns consumed: zero when slack is disabled, one for inequality
/// senses, two for equality (plus and minus). Columns are allocated sequentially
/// from `col_generic_slack_start` — plus first, then (for `==`) minus.
fn allocate_generic_slack_cols(
    constraint: &GenericConstraint,
    col_generic_slack_start: usize,
    n_slack_cols: &mut usize,
) -> (Option<usize>, Option<usize>) {
    if !constraint.slack.enabled {
        return (None, None);
    }
    let plus_col = col_generic_slack_start + *n_slack_cols;
    *n_slack_cols += 1;
    let minus_col = if constraint.sense == ConstraintSense::Equal {
        let mc = col_generic_slack_start + *n_slack_cols;
        *n_slack_cols += 1;
        Some(mc)
    } else {
        None
    };
    (Some(plus_col), minus_col)
}

/// Enumerate active generic constraint rows and assign their slack column indices.
///
/// For each active `(constraint, block)` pair at this stage, one
/// [`GenericConstraintRowEntry`] is produced — except a `block_id = None` bound
/// over a block-independent expression, which collapses to a single stage-level
/// row (see [`GenericConstraintRowEntry`]). Slack columns are allocated
/// sequentially from `col_generic_slack_start` — first the plus-slack, then
/// (for equality constraints) the minus-slack.
fn enumerate_generic_constraint_rows(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    n_blks: usize,
    col_generic_slack_start: usize,
) -> GenericConstraintLayout {
    let mut n_generic_rows: usize = 0;
    let mut n_generic_slack_cols: usize = 0;
    let mut generic_constraint_rows: Vec<GenericConstraintRowEntry> = Vec::new();

    for (constraint_idx, constraint) in ctx.generic_constraints.iter().enumerate() {
        if !ctx
            .resolved_generic_bounds
            .is_active(constraint_idx, stage.id)
        {
            continue;
        }

        let bound_entries = ctx
            .resolved_generic_bounds
            .bounds_for_stage(constraint_idx, stage.id);

        // A `block_id = None` bound over a block-independent expression produces
        // identical rows for every block, so it collapses to a single stage-level
        // row priced by the stage's total hours. Block-level expressions keep the
        // per-block replication (the rows differ by block).
        let collapse_stage_level =
            crate::generic_constraints::expression_is_block_independent(&constraint.expression);

        for &(block_id, bound) in bound_entries {
            match block_id {
                None if collapse_stage_level => {
                    // Single collapsed stage-level row (block_idx = 0 sentinel).
                    let (slack_plus_col, slack_minus_col) = allocate_generic_slack_cols(
                        constraint,
                        col_generic_slack_start,
                        &mut n_generic_slack_cols,
                    );
                    n_generic_rows += 1;
                    generic_constraint_rows.push(GenericConstraintRowEntry {
                        constraint_idx,
                        entity_id: constraint.id.0,
                        block_idx: 0,
                        is_stage_level: true,
                        bound,
                        sense: constraint.sense,
                        slack_enabled: constraint.slack.enabled,
                        slack_penalty: constraint.slack.penalty.unwrap_or(0.0),
                        slack_plus_col,
                        slack_minus_col,
                    });
                }
                None => {
                    // One row per block (block-level expression).
                    for block_idx in 0..n_blks {
                        let (slack_plus_col, slack_minus_col) = allocate_generic_slack_cols(
                            constraint,
                            col_generic_slack_start,
                            &mut n_generic_slack_cols,
                        );
                        n_generic_rows += 1;
                        generic_constraint_rows.push(GenericConstraintRowEntry {
                            constraint_idx,
                            entity_id: constraint.id.0,
                            block_idx,
                            is_stage_level: false,
                            bound,
                            sense: constraint.sense,
                            slack_enabled: constraint.slack.enabled,
                            slack_penalty: constraint.slack.penalty.unwrap_or(0.0),
                            slack_plus_col,
                            slack_minus_col,
                        });
                    }
                }
                Some(blk_id) => {
                    // One row for the specific block (0-indexed from the block_id value).
                    // block_id in bounds is a non-negative 0-indexed block position;
                    // upstream validation ensures it is non-negative.
                    #[allow(clippy::cast_sign_loss)]
                    let block_idx = blk_id as usize;
                    let (slack_plus_col, slack_minus_col) = allocate_generic_slack_cols(
                        constraint,
                        col_generic_slack_start,
                        &mut n_generic_slack_cols,
                    );
                    n_generic_rows += 1;
                    generic_constraint_rows.push(GenericConstraintRowEntry {
                        constraint_idx,
                        entity_id: constraint.id.0,
                        block_idx,
                        is_stage_level: false,
                        bound,
                        sense: constraint.sense,
                        slack_enabled: constraint.slack.enabled,
                        slack_penalty: constraint.slack.penalty.unwrap_or(0.0),
                        slack_plus_col,
                        slack_minus_col,
                    });
                }
            }
        }
    }

    GenericConstraintLayout {
        n_generic_rows,
        n_generic_slack_cols,
        generic_constraint_rows,
    }
}

impl StageLayout {
    // Rationale: single cohesive LP layout constructor; every local binding contributes to
    // the `Self { .. }` literal that terminates the function.  Each `StageLayout` field is
    // assigned next to the `idx` field (or NCS/generic/fishing arithmetic) it reads from, so
    // keeping the indexer-owned reads and the NCS/generic/fishing-only derivations in one
    // place is what makes the read-vs-recompute distinction auditable; splitting would scatter
    // the field initializers across helpers and obscure which offsets the indexer owns.
    #[allow(clippy::too_many_lines)]
    pub(crate) fn new(ctx: &TemplateBuildCtx<'_>, stage: &Stage, stage_idx: usize) -> Self {
        let n_blks = stage.blocks.len();

        // Identify FPHA and evaporation hydros before constructing the augmented
        // indexer, since their indices are needed for the indexer's `FphaColumnLayout`
        // and `EvapConfig` arguments. The indexer owns the resulting column offsets.
        let (fpha_hydro_indices, fpha_planes_per_hydro) = identify_fpha_hydros(ctx, stage_idx);
        let evap_hydro_indices = identify_evap_hydros(ctx);

        // Compute max_deficit_segments once; the indexer derives every offset
        // downstream of the deficit block from this count.
        let max_deficit_segments = ctx
            .buses
            .iter()
            .map(|b| b.deficit_segments.len())
            .max()
            .unwrap_or(0);

        // Build the augmented indexer with the real anticipated metadata.
        // When `n_anticipated == 0` the anticipated_state block is empty and the
        // layout is bit-identical to the base indexer without anticipated columns.
        let idx = StageIndexer::with_equipment_and_evaporation(
            &crate::indexer::EquipmentCounts {
                hydro_count: ctx.n_hydros,
                max_par_order: ctx.max_par_order,
                n_thermals: ctx.n_thermals,
                n_lines: ctx.n_lines,
                n_buses: ctx.n_buses,
                n_blks,
                has_inflow_penalty: ctx.has_penalty,
                max_deficit_segments,
                n_anticipated: ctx.n_anticipated,
                k_max: ctx.k_max,
                n_pumping: ctx.n_pumping,
                anticipated_lead_stages: ctx.anticipated_lead_stages.clone(),
                anticipated_thermal_indices: ctx.anticipated_thermal_indices.clone(),
            },
            &crate::indexer::FphaColumnLayout {
                hydro_indices: fpha_hydro_indices.clone(),
                planes_per_hydro: fpha_planes_per_hydro.clone(),
            },
            &crate::indexer::EvapConfig {
                hydro_indices: evap_hydro_indices.clone(),
            },
        );

        let n_ant_state = ctx.n_anticipated * ctx.k_max;

        // NCS: identify active entities and compute their column region.
        // The NCS block follows the last operational-violation slack family
        // (`generation_below_slack`), so its start is that family's end. The
        // generation-below slack is non-empty whenever `hydro_count > 0`; when
        // there are no hydros it is the empty `0..0` sentinel, so fall back to
        // the evaporation-column cursor (`evap_col_start`), which equals the end
        // of all pre-NCS columns when `n_hydros == 0` because the evaporation and
        // withdrawal/operational-slack blocks are then both empty.
        let active_ncs_indices = identify_active_ncs(ctx, stage);
        let n_active_ncs = active_ncs_indices.len();
        let col_ncs_start = if ctx.n_hydros > 0 {
            idx.generation_below_slack.end
        } else {
            idx.evap_col_start()
        };
        let col_ncs_end = col_ncs_start + n_active_ncs * n_blks;

        // FPHA generation and evaporation column starts: read from the indexer
        // accessors so the empty-block cursor (not the normalised `0..0`) is
        // used when no FPHA/evap hydros exist.
        let col_generation_start = idx.generation_col_start();
        let col_evap_start = idx.evap_col_start();

        // Withdrawal and operational-violation slack columns: non-empty whenever
        // `hydro_count > 0` and then carry the canonical cursors; with no hydros
        // they normalise to `0..0`, so fall back to the evaporation-column cursor
        // (`evap_col_start`), which is the correct column index for an empty
        // withdrawal/operational-slack region.
        let (
            col_withdrawal_neg_start,
            col_withdrawal_pos_start,
            col_outflow_below_start,
            col_outflow_above_start,
            col_turbine_below_start,
            col_generation_below_start,
        ) = if ctx.n_hydros > 0 {
            (
                idx.withdrawal_slack_neg.start,
                idx.withdrawal_slack_pos.start,
                idx.outflow_below_slack.start,
                idx.outflow_above_slack.start,
                idx.turbine_below_slack.start,
                idx.generation_below_slack.start,
            )
        } else {
            let region_start = idx.evap_col_start();
            (
                region_start,
                region_start,
                region_start,
                region_start,
                region_start,
                region_start,
            )
        };

        // Row offsets: z_inflow, water balance, load balance, FPHA, evap, operational, generic.
        // z_inflow starts at row 0; state pinning is applied via column bounds.
        // `n_state` from the augmented indexer is the column-side state dimension:
        // `N*(1+L) + n_anticipated*k_max`.
        let n_state = idx.n_state;
        // The dual-relevant structural prefix of view.dual is empty because state
        // pinning uses column bounds. Cut-subgradient extraction reads
        // view.reduced_costs; n_dual_relevant on the row side is unused by the cut path.
        let n_dual_relevant = 0_usize;
        // Leading row regions: read from the augmented indexer (the single owner
        // of this chain). The FPHA and evap row blocks normalise to empty, so
        // read their cursors via the indexer accessors rather than the `0..0`
        // public ranges: `row_fpha_start` is the load-balance row end, and
        // `row_evap_start` is the row at which evap rows begin (the FPHA-rows
        // end cursor). The operational-violation row blocks keep their cursor
        // even when empty, so their `.start` fields are read directly.
        let row_water_balance_start = idx.water_balance.start;
        let row_load_balance_start = idx.load_balance.start;
        let row_fpha_start = idx.load_balance.end;
        let row_evap_start = idx.fpha_rows_end();
        let n_op_rows = ctx.n_hydros * n_blks;
        // The four operational-violation row blocks are non-empty whenever
        // `hydro_count > 0` and then carry the canonical cursors; with no hydros
        // they normalise to `0..0`, so fall back to the evaporation-row end
        // cursor (`row_evap_start + n_evap_hydros`), which is what the empty
        // region begins at.
        let (
            row_min_outflow_start,
            row_max_outflow_start,
            row_min_turbine_start,
            row_min_generation_start,
        ) = if ctx.n_hydros > 0 {
            (
                idx.min_outflow_rows.start,
                idx.max_outflow_rows.start,
                idx.min_turbine_rows.start,
                idx.min_generation_rows.start,
            )
        } else {
            let evap_rows_end = row_evap_start + idx.n_evap_hydros;
            (evap_rows_end, evap_rows_end, evap_rows_end, evap_rows_end)
        };

        // One fishing row per anticipated plant at every stage (always-active).
        let n_anticipated_fishing_rows = ctx.n_anticipated;
        let row_anticipated_fishing_start = row_min_generation_start + n_op_rows;

        // Anticipated-state-out definition rows: one per ACTIVE plant (strict gate).
        // Active means stage_idx + K_p < n_stages (same predicate as
        // `anticipated_decision_active_at_stage`). Inactive plants emit no row.
        // Placed immediately after fishing rows, before generic rows.
        let n_stages = ctx.bounds.n_stages();
        let n_anticipated_state_out_def_rows = ctx
            .anticipated_lead_stages
            .iter()
            .filter(|&&k_i| stage_idx.saturating_add(k_i) < n_stages)
            .count();
        let row_anticipated_state_out_def_start =
            row_anticipated_fishing_start + n_anticipated_fishing_rows;
        let row_generic_start =
            row_anticipated_state_out_def_start + n_anticipated_state_out_def_rows;

        // Anticipated-state column start from the augmented indexer.
        // Slot-major layout: col for slot s, plant i = anticipated_state.start + s * n_anticipated + i.
        let col_anticipated_state_start = idx.anticipated_state.start;

        // Pumping-flow columns sit between the NCS region and the generic-slack
        // columns, block-major (`col_pumping_start + p * n_blks + blk`). The
        // station count is the ctx's authoritative `n_pumping` (the entity-slice
        // length), asserted equal to `bounds.n_pumping()` at ctx construction.
        // When `n_pumping == 0` the block is empty, so `col_pumping_end ==
        // col_ncs_end` and every downstream cursor (generic-slack, z-inflow,
        // `num_cols`) is unshifted.
        let n_pumping = ctx.n_pumping;
        let col_pumping_start = col_ncs_end;
        let col_pumping_end = col_pumping_start + n_pumping * n_blks;

        // Generic constraints: active rows and slack columns.
        let col_generic_slack_start = col_pumping_end;
        let generic =
            enumerate_generic_constraint_rows(ctx, stage, n_blks, col_generic_slack_start);

        // z-inflow columns and rows: positions from the augmented indexer.
        let col_z_inflow_start = idx.z_inflow.start;
        let row_z_inflow_start = idx.z_inflow_row_start;

        // Scalar layout offsets needed by matrix helpers — read from the
        // augmented indexer so they shift correctly when n_anticipated > 0.
        let col_theta = idx.theta;
        let col_storage_in_start = idx.storage_in.start;
        let col_inflow_lags_start = idx.inflow_lags.start;

        let num_cols = col_generic_slack_start + generic.n_generic_slack_cols;
        let num_rows = row_generic_start + generic.n_generic_rows;
        let zeta = stage.blocks.iter().map(|b| b.duration_hours).sum::<f64>() * M3S_TO_HM3;

        Self {
            n_blks,
            n_h: ctx.n_hydros,
            lag_order: ctx.max_par_order,
            n_anticipated: ctx.n_anticipated,
            k_max: ctx.k_max,
            n_ant_state,
            col_theta,
            col_storage_in_start,
            col_inflow_lags_start,
            col_turbine_start: idx.turbine.start,
            col_spillage_start: idx.spillage.start,
            col_diversion_start: idx.diversion.start,
            col_thermal_start: idx.thermal.start,
            // The anticipated blocks normalise to `0..0` when empty; their starts
            // are the thermal-block end plus the (possibly zero) anticipated count.
            col_anticipated_decision_start: idx.thermal.end,
            col_anticipated_state_out_start: idx.thermal.end + ctx.n_anticipated,
            col_anticipated_state_start,
            col_line_fwd_start: idx.line_fwd.start,
            col_line_rev_start: idx.line_rev.start,
            col_deficit_start: idx.deficit.start,
            max_deficit_segments: idx.max_deficit_segments,
            col_excess_start: idx.excess.start,
            // `inflow_slack` normalises to `0..0` without the penalty, so the
            // inflow-slack column start is the excess-block end cursor.
            col_inflow_slack_start: idx.excess.end,
            col_generation_start,
            col_evap_start,
            col_withdrawal_neg_start,
            col_withdrawal_pos_start,
            col_outflow_below_start,
            col_outflow_above_start,
            col_turbine_below_start,
            col_generation_below_start,
            col_ncs_start,
            n_ncs: n_active_ncs,
            active_ncs_indices,
            col_pumping_start,
            n_pumping,
            num_cols,
            row_water_balance_start,
            row_load_balance_start,
            row_fpha_start,
            row_evap_start,
            row_min_outflow_start,
            row_max_outflow_start,
            row_min_turbine_start,
            row_min_generation_start,
            // Permanent sentinel: state pinning uses column bounds, not rows.
            // This field is retained at 0 for API stability.
            row_anticipated_state_fixing_start: 0,
            row_anticipated_fishing_start,
            n_anticipated_fishing_rows,
            row_anticipated_state_out_def_start,
            n_anticipated_state_out_def_rows,
            row_generic_start,
            num_rows,
            n_generic_rows: generic.n_generic_rows,
            row_z_inflow_start,
            col_z_inflow_start,
            n_state,
            n_dual_relevant,
            zeta,
            fpha_hydro_indices,
            fpha_planes_per_hydro,
            evap_hydro_indices,
            generic_constraint_rows: generic.generic_constraint_rows,
            indexer: idx,
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines
)]
mod tests {
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
    use crate::indexer::StageIndexer;
    use crate::resolved_parameters::ResolvedParameters;

    use super::{StageLayout, TemplateBuildCtx};

    // ── Fixture helpers ───────────────────────────────────────────────────────

    /// Owns all data needed to construct a zero-entity `TemplateBuildCtx`.
    ///
    /// Fields are kept together so that references into them share a single
    /// lifetime `'_`, avoiding the 16-argument helper that clippy flags.
    struct ZeroEntityFixtures {
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

    impl ZeroEntityFixtures {
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

        /// Build a zero-entity `TemplateBuildCtx` with the supplied
        /// anticipated-metadata overrides.
        ///
        /// All slice fields are empty; all scalar entity counts are zero except
        /// the anticipated fields provided by the caller.
        fn make_ctx(
            &self,
            n_anticipated: usize,
            k_max: usize,
            anticipated_lead_stages: Vec<usize>,
            anticipated_thermal_indices: Vec<usize>,
        ) -> TemplateBuildCtx<'_> {
            TemplateBuildCtx {
                hydros: &[],
                thermals: &[],
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                bounds: &self.bounds,
                penalties: &self.penalties,
                hydro_pos: HashMap::new(),
                thermal_pos: HashMap::new(),
                line_pos: HashMap::new(),
                bus_pos: HashMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                resolved_generic_bounds: &self.resolved_generic_bounds,
                resolved_load_factors: &self.resolved_load_factors,
                resolved_exchange_factors: &self.resolved_exchange_factors,
                non_controllable_sources: &[],
                resolved_ncs_bounds: &self.resolved_ncs_bounds,
                resolved_ncs_factors: &self.resolved_ncs_factors,
                pumping_stations: &[],
                pumping_pos: HashMap::new(),
                n_pumping: 0,
                resolved_parameters: &self.resolved_parameters,
                diversion_upstream: HashMap::new(),
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated,
                k_max,
                anticipated_lead_stages,
                anticipated_thermal_indices,
                has_penalty: false,
                // Tests that use ZeroEntityFixtures don't exercise discount
                // factors; provide n_stages = 1 element vecs that won't panic.
                cumulative_discount_factors: vec![1.0],
                total_hours_per_stage: vec![744.0],
            }
        }
    }

    /// Build a minimal `Stage` with one block of 744 hours.
    fn minimal_stage() -> Stage {
        Stage {
            index: 0,
            id: 0,
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
        }
    }

    // ── AC-3 ─────────────────────────────────────────────────────────────────

    /// AC-3: `StageLayout` built from a context with `n_anticipated == 0` has
    /// `n_ant_state == 0`, `n_anticipated == 0`, `k_max == 0`, and
    /// `col_turbine_start == idx.theta + 1` where `idx` is the legacy
    /// `StageIndexer::new(0, 0)` (zero hydros, zero lag order).
    ///
    /// This verifies that the decision-region offset before the
    /// `anticipated_state_out` insertion is preserved when no anticipated
    /// thermals are present.
    #[test]
    fn stage_layout_zero_anticipated_matches_pre_anticipated_offsets() {
        let fixtures = ZeroEntityFixtures::new();
        let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
        let stage = minimal_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);

        // n_ant_state, n_anticipated, k_max must all be zero.
        assert_eq!(layout.n_ant_state, 0, "n_ant_state");
        assert_eq!(layout.n_anticipated, 0, "n_anticipated");
        assert_eq!(layout.k_max, 0, "k_max");

        // col_turbine_start must equal the legacy theta + 1.
        let idx = StageIndexer::new(ctx.n_hydros, ctx.max_par_order);
        assert_eq!(
            layout.col_turbine_start,
            idx.theta + 1,
            "col_turbine_start must equal idx.theta + 1 with zero anticipated"
        );
    }

    // ── Anticipated-decision column positioning ──────────────────────────────

    /// `col_anticipated_decision_start` falls between thermal end and
    /// `col_line_fwd_start` when `n_anticipated=2, n_thermals=3, n_blks=4`.
    ///
    /// The layout in the control region is:
    /// `thermal | anticipated_decision (2 cols) | anticipated_state_out (2 cols) | line_fwd`
    /// So `col_line_fwd_start == col_anticipated_decision_start + 2 * n_anticipated`.
    #[test]
    fn anticipated_decision_columns_placed_between_thermal_and_line_fwd() {
        use chrono::NaiveDate;
        use cobre_core::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
        };

        let fixtures = ZeroEntityFixtures::new();
        // ZeroEntityFixtures builds n_thermals=0, so the thermal per-block block is
        // empty and col_anticipated_decision_start == col_thermal_start. The two
        // stage-level anticipated blocks then separate col_thermal_start from
        // col_line_fwd_start by exactly 2 * n_anticipated columns.
        let n_anticipated = 2_usize;
        let k_max = 1_usize;
        let ctx = fixtures.make_ctx(n_anticipated, k_max, vec![1, 1], vec![0, 0]);

        let stage = Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![
                Block {
                    index: 0,
                    name: "B0".to_string(),
                    duration_hours: 186.0,
                },
                Block {
                    index: 1,
                    name: "B1".to_string(),
                    duration_hours: 186.0,
                },
                Block {
                    index: 2,
                    name: "B2".to_string(),
                    duration_hours: 186.0,
                },
                Block {
                    index: 3,
                    name: "B3".to_string(),
                    duration_hours: 186.0,
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
        };
        let layout = StageLayout::new(&ctx, &stage, 0);

        // n_thermals = 0, n_blks = 4:
        // col_thermal_start = col_diversion_start + 0 * 4 = col_diversion_start
        // col_anticipated_decision_start = col_thermal_start + 0 * 4 = col_thermal_start
        // col_anticipated_state_out_start = col_anticipated_decision_start + n_anticipated
        // col_line_fwd_start = col_anticipated_state_out_start + n_anticipated
        //                    = col_anticipated_decision_start + 2 * n_anticipated
        assert_eq!(
            layout.col_anticipated_decision_start, layout.col_thermal_start,
            "col_anticipated_decision_start must equal col_thermal_start \
             when n_thermals=0 (no thermal per-block cols)"
        );
        assert_eq!(
            layout.col_anticipated_state_out_start,
            layout.col_anticipated_decision_start + n_anticipated,
            "col_anticipated_state_out_start == col_anticipated_decision_start + n_anticipated"
        );
        assert_eq!(
            layout.col_line_fwd_start,
            layout.col_anticipated_state_out_start + n_anticipated,
            "col_line_fwd_start == col_anticipated_state_out_start + n_anticipated"
        );
        // Verify the separation between thermal_start and line_fwd_start is exactly 2*n_anticipated
        // (n_anticipated cols for anticipated_decision + n_anticipated cols for anticipated_state_out).
        assert_eq!(
            layout.col_line_fwd_start - layout.col_thermal_start,
            2 * n_anticipated,
            "gap from thermal_start to line_fwd_start must be exactly 2*n_anticipated (two stage-level blocks)"
        );
    }

    // ── AC-4 ─────────────────────────────────────────────────────────────────

    /// AC-4: `StageLayout` with `n_anticipated=2, k_max=3, n_hydros=0,
    /// max_par_order=0` has `col_turbine_start == 0*(3+0) + 6 + 1 == 7`.
    ///
    /// `n_ant_state = n_anticipated * k_max = 2 * 3 = 6` shifts `theta`
    /// from the legacy `N*(3+L) = 0` to `0 + 6 = 6`, so decisions begin at 7.
    ///
    /// The general formula (any N, L) is `N*(3+L) + n_ant_state + 1`.
    #[test]
    fn stage_layout_with_anticipated_shifts_decision_region() {
        let n_hydros = 0_usize;
        let max_par_order = 0_usize;
        let n_anticipated = 2_usize;
        let k_max = 3_usize;

        let fixtures = ZeroEntityFixtures::new();
        let ctx = fixtures.make_ctx(
            n_anticipated,
            k_max,
            vec![2, 3], // anticipated_lead_stages
            vec![0, 2], // anticipated_thermal_indices (arbitrary; layout doesn't inspect them)
        );
        let stage = minimal_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);

        // n_ant_state = n_anticipated * k_max = 2 * 3 = 6
        let expected_n_ant_state = n_anticipated * k_max;
        assert_eq!(layout.n_ant_state, expected_n_ant_state, "n_ant_state");

        // theta = N*(3+L) + n_ant_state = 0*(3+0) + 6 = 6
        // col_turbine_start = theta + 1 = 7
        let expected_col_turbine_start = n_hydros * (3 + max_par_order) + expected_n_ant_state + 1;
        assert_eq!(
            layout.col_turbine_start, expected_col_turbine_start,
            "col_turbine_start == N*(3+L) + n_ant_state + 1"
        );
    }

    // ── Anticipated-fishing row positioning ──────────────────────────────────

    /// `row_anticipated_fishing_start` immediately follows the operational
    /// violation row block, i.e. equals `row_min_generation_start + n_op_rows`.
    ///
    /// Uses a zero-hydro context so `n_op_rows == 0`, which means the fishing
    /// start equals `row_min_generation_start` exactly. The algebraic identity
    /// `row_anticipated_fishing_start == row_min_generation_start + n_op_rows`
    /// is verified for the general formula; the case `n_op_rows > 0` is covered by
    /// the production code path (`n_hydros * n_blks` counts operational violation rows).
    ///
    /// Setup: `n_anticipated=2`, `k_max=2`, `anticipated_lead_stages=[1,2]`,
    /// zero hydros, one block. At `stage_idx=1`:
    /// - `n_op_rows = 0 * 1 = 0` (no hydros)
    /// - `n_anticipated_fishing_rows = 1` (`K_0=1<=1` active, `K_1=2>1` inactive)
    /// - `row_anticipated_fishing_start` must equal `row_min_generation_start + 0`
    #[test]
    fn anticipated_fishing_row_offset_after_operational_violations() {
        let n_anticipated = 2_usize;
        let k_max = 2_usize;

        let fixtures = ZeroEntityFixtures::new();
        let ctx = fixtures.make_ctx(
            n_anticipated,
            k_max,
            vec![1, 2], // K_0=1, K_1=2
            vec![0, 1], // arbitrary thermal indices
        );
        let stage = minimal_stage(); // 1 block
        // stage_idx=1: always-active → both plants active → 2 fishing rows.
        let layout = StageLayout::new(&ctx, &stage, 1);

        // n_op_rows = n_hydros * n_blks = 0 * 1 = 0
        let n_op_rows = 0_usize;
        assert_eq!(
            layout.row_anticipated_fishing_start,
            layout.row_min_generation_start + n_op_rows,
            "row_anticipated_fishing_start must equal row_min_generation_start + n_op_rows"
        );
        // Always-active: both plants active at every stage → 2 fishing rows.
        assert_eq!(
            layout.n_anticipated_fishing_rows, 2,
            "n_anticipated_fishing_rows must equal n_anticipated (2) under always-active predicate"
        );
    }

    /// `n_anticipated_fishing_rows` equals `n_anticipated` at every stage under
    /// the always-active predicate. With `K_i=[1,2]` and `n_anticipated=2`, the
    /// count is 2 at every stage in `[0, 1, 2, 3]`.
    #[test]
    fn anticipated_fishing_row_count_grows_with_stage() {
        let n_anticipated = 2_usize;
        let k_max = 2_usize;

        let fixtures = ZeroEntityFixtures::new();
        let ctx = fixtures.make_ctx(
            n_anticipated,
            k_max,
            vec![1, 2], // K_0=1, K_1=2
            vec![0, 1], // arbitrary thermal indices
        );
        let stage = minimal_stage(); // 1 block

        for (stage_idx, expected) in [(0_usize, 2), (1, 2), (2, 2), (3, 2)] {
            let layout = StageLayout::new(&ctx, &stage, stage_idx);
            assert_eq!(
                layout.n_anticipated_fishing_rows, expected,
                "n_anticipated_fishing_rows must equal {expected} at stage_idx={stage_idx}"
            );
        }
    }

    /// `row_anticipated_state_fixing_start` is always the sentinel value 0.
    ///
    /// State pinning uses column bounds; the field always equals 0 regardless of
    /// `n_anticipated` or `k_max`. It is retained as a permanent sentinel for API stability.
    #[test]
    fn row_anticipated_state_fixing_start_equals_anticipated_state_column_start_numerically() {
        let n_anticipated = 2_usize;
        let k_max = 3_usize;

        let fixtures = ZeroEntityFixtures::new();
        let ctx = fixtures.make_ctx(
            n_anticipated,
            k_max,
            vec![3, 2], // K_0=3, K_1=2 (k_max=3 comes from K_0=3)
            vec![0, 1],
        );
        let stage = minimal_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);

        // row_anticipated_state_fixing_start is a permanent sentinel: state pinning
        // uses column bounds, so no state-fixing rows exist in the LP.
        assert_eq!(
            layout.row_anticipated_state_fixing_start, 0,
            "row_anticipated_state_fixing_start must be 0 (permanent sentinel)"
        );
        // col_anticipated_state_start is unchanged: still N*(1+L) = 0 for N=0.
        assert_eq!(
            layout.col_anticipated_state_start, 0,
            "col_anticipated_state_start must be 0 for N=0"
        );

        // num_rows in this fixture: n_state = N*(1+L) + A*K = 0 + 2*3 = 6
        // (lifted into anticipated_state via the augmented indexer). The
        // current row layout starts the first non-state block at row
        // `ctx.n_hydros` (== 0 here). The structural invariant we assert
        // is `row_water_balance_start == ctx.n_hydros` (no n_state offset;
        // state pinning is via column bounds, not rows).
        assert_eq!(
            layout.row_water_balance_start, ctx.n_hydros,
            "row_water_balance_start must equal ctx.n_hydros (the n_state offset is gone)"
        );
    }

    /// `num_rows` does not include state-fixing rows; the LP row layout starts
    /// directly with `z_inflow_rows` at row 0.
    ///
    /// State pinning uses column bounds, so the `[0, n_state)` row prefix
    /// from the pre-cutover layout is absent. `num_rows` equals the count of
    /// structural rows only (`z_inflow`, water balance, load balance, FPHA,
    /// evap, operational, fishing, `anticipated_state_out_def`, generic).
    #[test]
    fn num_rows_drops_by_n_state_with_anticipated_thermals() {
        let n_anticipated = 2_usize;
        let k_max = 3_usize;

        let fixtures = ZeroEntityFixtures::new();
        let ctx = fixtures.make_ctx(n_anticipated, k_max, vec![3, 2], vec![0, 1]);
        let stage = minimal_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);

        // n_state for this fixture: N*(1+L) + A*K = 0 + 2*3 = 6.
        let n_state = ctx.n_hydros * (1 + ctx.max_par_order) + n_anticipated * k_max;
        assert_eq!(n_state, 6);

        // Post-ticket num_rows for this zero-hydro fixture: only the
        // anticipated_fishing block contributes (2 active plants at stage 0).
        // All other row blocks are 0 (no hydros, no buses, no FPHA, no evap).
        let observed = layout.num_rows;
        assert_eq!(
            observed, 2,
            "post-ticket num_rows equals anticipated_fishing_rows (2) for this fixture"
        );

        // Reference value: if state-fixing rows were present, num_rows would be observed + n_state.
        let pre_ticket_expected = observed + n_state;
        assert_eq!(
            pre_ticket_expected, 8,
            "pre-ticket reference value (observed + n_state) is 8 for this fixture"
        );
        // Structural invariant proving the reduction: row_water_balance_start
        // equals ctx.n_hydros (no n_state offset). Pre-ticket it would have
        // been n_state + ctx.n_hydros.
        assert_eq!(
            layout.row_water_balance_start, ctx.n_hydros,
            "row_water_balance_start no longer includes the n_state offset"
        );
    }

    // ── Anticipated-decision range tests ──────────────────────────────────────

    /// Build a `ResolvedBounds` with zero entities but the given `n_stages`.
    ///
    /// Used to exercise the `stage_idx.saturating_add(k_i) < n_stages` predicate
    /// in `n_anticipated_state_out_def_rows` without needing real entity data.
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

    /// Builds a fixture struct owning all data for a context with anticipated
    /// thermals and a known `n_stages` for the `state_out_def` predicate.
    struct AntFixturesWithNStages {
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

    impl AntFixturesWithNStages {
        fn new(n_stages: usize) -> Self {
            Self {
                par_lp: PrecomputedPar::default(),
                cascade: CascadeTopology::build(&[]),
                bounds: bounds_with_n_stages(n_stages),
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
        ) -> TemplateBuildCtx<'_> {
            let n_stages = self.bounds.n_stages();
            TemplateBuildCtx {
                hydros: &[],
                thermals: &[],
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                bounds: &self.bounds,
                penalties: &self.penalties,
                hydro_pos: HashMap::new(),
                thermal_pos: HashMap::new(),
                line_pos: HashMap::new(),
                bus_pos: HashMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                resolved_generic_bounds: &self.resolved_generic_bounds,
                resolved_load_factors: &self.resolved_load_factors,
                resolved_exchange_factors: &self.resolved_exchange_factors,
                non_controllable_sources: &[],
                resolved_ncs_bounds: &self.resolved_ncs_bounds,
                resolved_ncs_factors: &self.resolved_ncs_factors,
                pumping_stations: &[],
                pumping_pos: HashMap::new(),
                n_pumping: 0,
                resolved_parameters: &self.resolved_parameters,
                diversion_upstream: HashMap::new(),
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated,
                k_max,
                anticipated_lead_stages,
                anticipated_thermal_indices,
                has_penalty: false,
                cumulative_discount_factors: vec![1.0; n_stages],
                total_hours_per_stage: vec![744.0; n_stages],
            }
        }
    }

    /// `col_anticipated_state_out_start` is adjacent to `col_anticipated_decision_start`,
    /// `col_line_fwd_start` follows `col_anticipated_state_out_start`, and
    /// `n_anticipated_state_out_def_rows` counts both active plants at stage 0.
    ///
    /// Fixture: `n_anticipated=2`, `K=[2,3]`, `n_stages=6`, `stage_idx=0`.
    /// Both plants are active: `0+2=2 < 6` and `0+3=3 < 6`.
    #[test]
    fn test_layout_state_out_block_adjacent_to_decision() {
        let fixtures = AntFixturesWithNStages::new(6);
        let ctx = fixtures.make_ctx(
            2,          // n_anticipated
            3,          // k_max
            vec![2, 3], // K_0=2, K_1=3
            vec![0, 1],
        );
        let stage = minimal_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);

        assert_eq!(
            layout.col_anticipated_state_out_start,
            layout.col_anticipated_decision_start + 2,
            "state_out columns must be immediately after anticipated_decision"
        );
        assert_eq!(
            layout.col_line_fwd_start,
            layout.col_anticipated_state_out_start + 2,
            "line_fwd must be immediately after state_out columns"
        );
        assert_eq!(layout.n_anticipated_state_out_def_rows, 2);
        assert_eq!(
            layout.row_anticipated_state_out_def_start,
            layout.row_anticipated_fishing_start + layout.n_anticipated_fishing_rows
        );
    }

    /// `n_anticipated_state_out_def_rows == 0` when all plants are inactive at
    /// the given stage, but the column block stays allocated.
    ///
    /// Fixture: `n_anticipated=2`, `K=[2,3]`, `n_stages=6`, `stage_idx=5`.
    /// Both inactive: `5+2=7 >= 6` and `5+3=8 >= 6`.
    #[test]
    fn test_layout_state_out_def_rows_zero_when_all_inactive() {
        let fixtures = AntFixturesWithNStages::new(6);
        let ctx = fixtures.make_ctx(
            2,          // n_anticipated
            3,          // k_max
            vec![2, 3], // K_0=2, K_1=3
            vec![0, 1],
        );
        let stage = minimal_stage();
        let layout = StageLayout::new(&ctx, &stage, 5);

        assert_eq!(layout.n_anticipated_state_out_def_rows, 0);
        // Column block stays allocated regardless of activity.
        assert_eq!(
            layout.col_anticipated_state_out_start,
            layout.col_anticipated_decision_start + 2
        );
    }

    /// Zero-anticipated layouts must not grow `num_cols` or emit def rows.
    ///
    /// `col_anticipated_state_out_start` must equal `col_anticipated_decision_start`
    /// when `n_anticipated == 0` (empty block; both starts coincide).
    #[test]
    fn test_layout_no_anticipated_unchanged_num_cols() {
        let fixtures = ZeroEntityFixtures::new();
        let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
        let stage = minimal_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);

        assert_eq!(
            layout.col_anticipated_state_out_start, layout.col_anticipated_decision_start,
            "col_anticipated_state_out_start must equal col_anticipated_decision_start when n_anticipated=0"
        );
        assert_eq!(layout.n_anticipated_state_out_def_rows, 0);
    }

    // ── Pumping-flow column region ─────────────────────────────────────────────

    /// Build a `ResolvedBounds` with the given pumping-station count and stage
    /// count (all other entity tables empty). `table.n_pumping()` recovers
    /// `n_pumping` from the `pumping` Vec length divided by `n_stages`.
    fn bounds_with_pumping(n_pumping: usize, n_stages: usize) -> ResolvedBounds {
        ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_pumping,
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

    /// Owns the data for a `TemplateBuildCtx` whose `bounds` report a non-zero
    /// `n_pumping()`. Mirrors `ZeroEntityFixtures` but injects a pumping-aware
    /// `ResolvedBounds` so `StageLayout::new` reserves the `pumping_flow` block.
    struct PumpingFixtures {
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

    impl PumpingFixtures {
        fn new(n_pumping: usize, n_stages: usize) -> Self {
            Self {
                par_lp: PrecomputedPar::default(),
                cascade: CascadeTopology::build(&[]),
                bounds: bounds_with_pumping(n_pumping, n_stages),
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

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
            let n_stages = self.bounds.n_stages();
            TemplateBuildCtx {
                hydros: &[],
                thermals: &[],
                lines: &[],
                buses: &[],
                load_models: &[],
                cascade: &self.cascade,
                bounds: &self.bounds,
                penalties: &self.penalties,
                hydro_pos: HashMap::new(),
                thermal_pos: HashMap::new(),
                line_pos: HashMap::new(),
                bus_pos: HashMap::new(),
                par_lp: &self.par_lp,
                production_models: &self.production_models,
                evaporation_models: &self.evaporation_models,
                generic_constraints: &[],
                resolved_generic_bounds: &self.resolved_generic_bounds,
                resolved_load_factors: &self.resolved_load_factors,
                resolved_exchange_factors: &self.resolved_exchange_factors,
                non_controllable_sources: &[],
                resolved_ncs_bounds: &self.resolved_ncs_bounds,
                resolved_ncs_factors: &self.resolved_ncs_factors,
                // This fixture probes only the column-reservation arithmetic, which
                // consumes `n_pumping` (not the entity slice). The station slice is
                // left empty while `n_pumping` is sourced from the pumping-aware
                // `ResolvedBounds` so `StageLayout::new` reserves the right block;
                // the slice/`pumping_pos` threading is covered by the
                // `build_template_build_ctx` tests in `template.rs`.
                pumping_stations: &[],
                pumping_pos: HashMap::new(),
                n_pumping: self.bounds.n_pumping(),
                resolved_parameters: &self.resolved_parameters,
                diversion_upstream: HashMap::new(),
                n_hydros: 0,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 0,
                max_par_order: 0,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                has_penalty: false,
                cumulative_discount_factors: vec![1.0; n_stages],
                total_hours_per_stage: vec![744.0; n_stages],
            }
        }

        /// Build a stage with `n_blks` equal-duration blocks.
        fn stage_with_blocks(n_blks: usize) -> Stage {
            use cobre_core::Block;
            Stage {
                index: 0,
                id: 0,
                start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
                season_id: Some(0),
                blocks: (0..n_blks)
                    .map(|b| Block {
                        index: b,
                        name: format!("B{b}"),
                        duration_hours: 248.0,
                    })
                    .collect(),
                block_mode: cobre_core::BlockMode::Parallel,
                state_config: cobre_core::StageStateConfig {
                    storage: false,
                    inflow_lags: false,
                },
                risk_config: cobre_core::StageRiskConfig::Expectation,
                scenario_config: cobre_core::ScenarioSourceConfig {
                    branching_factor: 1,
                    noise_method: cobre_core::NoiseMethod::Saa,
                },
            }
        }
    }

    /// Inert-layout invariant: with `n_pumping == 0` the `pumping_flow` block
    /// collapses, so `col_pumping_start` sits exactly where the generic-slack
    /// columns begin (`col_ncs_end`, which equals `col_ncs_start` when no NCS are
    /// active) and `num_cols` is unshifted. For this zero-entity one-block system
    /// the entire column count is the single theta column.
    ///
    /// Pinning `n_pumping == 0`, `col_pumping_start == col_ncs_start`, and the
    /// exact `num_cols`/equipment starts proves that reserving the pumping region
    /// does not move any pre-existing column when there are no stations.
    #[test]
    fn pumping_layout_inert_when_no_stations() {
        let fixtures = ZeroEntityFixtures::new();
        let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
        let stage = minimal_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);

        // No stations: the bounds table reports zero pumping.
        assert_eq!(ctx.bounds.n_pumping(), 0, "fixture has no pumping stations");
        assert_eq!(layout.n_pumping, 0, "layout.n_pumping must be 0");

        // The empty pumping block does not advance the cursor: its start equals
        // the NCS-region end. With zero active NCS, col_ncs_end == col_ncs_start.
        assert_eq!(
            layout.col_pumping_start, layout.col_ncs_start,
            "col_pumping_start must equal col_ncs_start (col_ncs_end) when no stations"
        );

        // Pre-existing column starts for the zero-entity, single-block layout:
        // theta == 0, every equipment/slack/NCS region empty starting at theta+1.
        let idx = StageIndexer::new(ctx.n_hydros, ctx.max_par_order);
        let expected_start = idx.theta + 1;
        assert_eq!(layout.col_turbine_start, expected_start);
        assert_eq!(layout.col_thermal_start, expected_start);
        assert_eq!(layout.col_line_fwd_start, expected_start);
        assert_eq!(layout.col_deficit_start, expected_start);
        assert_eq!(layout.col_excess_start, expected_start);
        assert_eq!(layout.col_ncs_start, expected_start);
        assert_eq!(layout.col_pumping_start, expected_start);
        // num_cols is the single theta column; the empty pumping block adds nothing.
        assert_eq!(
            layout.num_cols, expected_start,
            "num_cols must be unshifted"
        );
    }

    /// `n_pumping == 2`, `n_blks == 3` ⇒ a 6-column `pumping_flow` block at
    /// `col_pumping_start`, block-major, and `num_cols` increased by exactly 6
    /// relative to the otherwise-identical station-free layout.
    #[test]
    fn pumping_layout_reserves_block_major_columns() {
        let n_pumping = 2_usize;
        let n_blks = 3_usize;

        // Baseline: identical zero-entity 3-block layout with no stations.
        let baseline_fixtures = PumpingFixtures::new(0, 3);
        let baseline_ctx = baseline_fixtures.make_ctx();
        let stage = PumpingFixtures::stage_with_blocks(n_blks);
        let baseline = StageLayout::new(&baseline_ctx, &stage, 0);
        assert_eq!(baseline.n_pumping, 0);

        // Station-bearing layout: 2 pumping stations across 3 stages.
        let fixtures = PumpingFixtures::new(n_pumping, 3);
        let ctx = fixtures.make_ctx();
        assert_eq!(
            ctx.bounds.n_pumping(),
            n_pumping,
            "fixture bounds must report n_pumping() == 2"
        );
        let layout = StageLayout::new(&ctx, &stage, 0);

        assert_eq!(layout.n_pumping, n_pumping, "layout.n_pumping == 2");
        // The pumping block begins exactly where the NCS region ends (no NCS here).
        assert_eq!(
            layout.col_pumping_start, layout.col_ncs_start,
            "col_pumping_start must follow the NCS region"
        );
        // Block-major width: n_pumping * n_blks == 2 * 3 == 6.
        assert_eq!(
            layout.num_cols - baseline.num_cols,
            n_pumping * n_blks,
            "num_cols must grow by exactly n_pumping * n_blks == 6"
        );
        // The 6 reserved columns occupy [col_pumping_start, col_pumping_start + 6).
        assert_eq!(
            layout.num_cols,
            layout.col_pumping_start + n_pumping * n_blks,
            "the 6-column block ends at num_cols (no generic-slack columns here)"
        );
    }
}
