use std::collections::{BTreeMap, HashMap};

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

use super::{
    EVAP_COLS_PER_HYDRO, EVAP_F_MINUS_OFFSET, EVAP_F_PLUS_OFFSET, EVAP_FLOW_OFFSET,
    GenericConstraintRowEntry, M3S_TO_HM3,
};

/// Pre-resolved bound, penalty, and factor tables shared across all stages.
///
/// Groups the eight `&'a` references that share the "resolved at setup, read
/// per stage" role into one cohesive concern, nested under
/// `TemplateBuildCtx::resolved`. Every field is a borrow with the same lifetime
/// `'a` as [`TemplateBuildCtx`]; grouping changes only struct shape, not any
/// value, ownership, or the column/row iteration order the `fill_*` matrix
/// helpers and generic-constraint resolvers visit. Read through
/// `ctx.resolved.<field>`.
pub(crate) struct ResolvedTables<'a> {
    pub(crate) bounds: &'a ResolvedBounds,
    pub(crate) penalties: &'a ResolvedPenalties,
    /// Pre-resolved table mapping `(constraint_idx, stage_id)` to active bound entries.
    pub(crate) resolved_generic_bounds: &'a ResolvedGenericConstraintBounds,
    /// Pre-resolved per-block load scaling factors.
    pub(crate) resolved_load_factors: &'a ResolvedLoadFactors,
    /// Pre-resolved per-block exchange capacity factors.
    pub(crate) resolved_exchange_factors: &'a ResolvedExchangeFactors,
    /// Pre-resolved per-stage NCS available generation bounds.
    pub(crate) resolved_ncs_bounds: &'a ResolvedNcsBounds,
    /// Pre-resolved per-block NCS generation scaling factors.
    pub(crate) resolved_ncs_factors: &'a ResolvedNcsFactors,
    /// Lookup table for parameter coefficient resolution.
    ///
    /// Maps `(parameter_id, stage_idx)` to a pre-resolved `f64` value.
    /// Queried by the LP builder when a [`cobre_core::CoefficientRef::Parameter`]
    /// term is encountered in a generic constraint expression.
    pub(crate) resolved_parameters: &'a crate::resolved_parameters::ResolvedParameters,
}

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
    /// Pre-resolved bound, penalty, and factor tables (see [`ResolvedTables`]).
    pub(crate) resolved: ResolvedTables<'a>,
    /// Entity-id → canonical slot index. `BTreeMap`, not `HashMap`: the maps are
    /// read by keyed `.get()` today, but the ordered-iteration guarantee makes
    /// declaration-order bit-determinism structural — an accidental iterating
    /// fill emits entries in `EntityId` order (the canonical `id.0` slot order)
    /// instead of nondeterministic `HashMap` order. Certified by
    /// `csc_byte_identical_under_permuted_multi_entity_order`.
    pub(crate) hydro_pos: BTreeMap<EntityId, usize>,
    pub(crate) thermal_pos: BTreeMap<EntityId, usize>,
    pub(crate) line_pos: BTreeMap<EntityId, usize>,
    pub(crate) bus_pos: BTreeMap<EntityId, usize>,
    pub(crate) par_lp: &'a PrecomputedPar,
    /// Resolved production models for all (hydro, stage) pairs.
    pub(crate) production_models: &'a ProductionModelSet,
    /// Resolved evaporation models for all hydro plants.
    pub(crate) evaporation_models: &'a EvaporationModelSet,
    /// Generic constraint definitions (expression, sense, slack config).
    pub(crate) generic_constraints: &'a [GenericConstraint],
    /// Non-controllable source entities sorted by ID.
    pub(crate) non_controllable_sources: &'a [NonControllableSource],
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
    pub(crate) pumping_pos: BTreeMap<EntityId, usize>,
    /// Number of pumping stations (`pumping_stations.len()`).
    ///
    /// The authoritative count threaded into `EquipmentCounts.n_pumping`, so the
    /// layout reserves `n_pumping * n_blks` pumping-flow columns. Asserted equal
    /// to `bounds.n_pumping()` at ctx construction — a divergence means the
    /// resolved-bounds table and the entity slice disagree on station count.
    pub(crate) n_pumping: usize,
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

/// Column/row offsets describing the anticipated-thermal layout for one stage.
///
/// Groups the seven offsets that together address the anticipated-decision,
/// anticipated-state-out, and anticipated-fishing column/row blocks into one
/// cohesive concern, nested under `StageLayout::anticipated`. Built from
/// existing `StageLayout::new` bindings; nesting changes only struct shape, not
/// any value or the column/row iteration order the `fill_anticipated_*` helpers
/// visit.
pub(crate) struct AnticipatedLayout {
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
    // Column regions
    /// Anticipated-thermal column/row offsets grouped into one cohesive concern.
    ///
    /// Holds the anticipated-decision / state-out / state / fishing column and
    /// row starts (see [`AnticipatedLayout`]). Read by the `fill_anticipated_*`
    /// matrix helpers through `layout.anticipated.<field>`.
    pub(crate) anticipated: AnticipatedLayout,
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
    // Template metadata
    pub(crate) n_dual_relevant: usize,
    // Scalar derived quantities used by row-bound and matrix helpers
    pub(crate) zeta: f64,
    // FPHA hydro information for this stage
    /// Indices (into `ctx.hydros`) of hydros using FPHA at this stage.
    pub(crate) fpha_hydro_indices: Vec<usize>,
    /// Inverse of `fpha_hydro_indices`: system hydro index → FPHA-local index.
    ///
    /// Length `n_h`. `Some(local_fpha_idx)` at each FPHA hydro's system index,
    /// `None` at every non-FPHA hydro. Single owner of the system→FPHA-local
    /// reverse map, read by the `fill_load_balance_entries` and
    /// `fill_operational_violation_entries` matrix helpers in place of rebuilding
    /// the same `Vec<Option<usize>>` inline at each call.
    pub(crate) fpha_local_index: Vec<Option<usize>>,
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
            .resolved
            .resolved_generic_bounds
            .is_active(constraint_idx, stage.id)
        {
            continue;
        }

        let bound_entries = ctx
            .resolved
            .resolved_generic_bounds
            .bounds_for_stage(constraint_idx, stage.id);

        // A `block_id = None` bound over a block-independent expression produces
        // identical rows for every block, so it collapses to a single stage-level
        // row priced by the stage's total hours. Block-level expressions keep the
        // per-block replication (the rows differ by block).
        let collapse_stage_level =
            crate::generic_constraints::expression_is_block_independent(&constraint.expression);

        // The constraint-invariant fields are identical across every row this
        // constraint produces; only `block_idx`, `is_stage_level`, the bound, and
        // the freshly-allocated slack columns vary per row. Bind the invariants and
        // build each entry through this closure so the three arms below stay
        // field-for-field identical.
        let entity_id = constraint.id.0;
        let sense = constraint.sense;
        let slack_enabled = constraint.slack.enabled;
        let slack_penalty = constraint.slack.penalty.unwrap_or(0.0);
        let make_entry = |block_idx: usize,
                          is_stage_level: bool,
                          slack_plus_col: Option<usize>,
                          slack_minus_col: Option<usize>,
                          bound: f64| {
            GenericConstraintRowEntry {
                constraint_idx,
                entity_id,
                block_idx,
                is_stage_level,
                bound,
                sense,
                slack_enabled,
                slack_penalty,
                slack_plus_col,
                slack_minus_col,
            }
        };

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
                    generic_constraint_rows.push(make_entry(
                        0,
                        true,
                        slack_plus_col,
                        slack_minus_col,
                        bound,
                    ));
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
                        generic_constraint_rows.push(make_entry(
                            block_idx,
                            false,
                            slack_plus_col,
                            slack_minus_col,
                            bound,
                        ));
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
                    generic_constraint_rows.push(make_entry(
                        block_idx,
                        false,
                        slack_plus_col,
                        slack_minus_col,
                        bound,
                    ));
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

        // Inverse of `fpha_hydro_indices`: system hydro index → FPHA-local index.
        // Built once here so the matrix-fill helpers read the cached map instead of
        // reconstructing the same `Vec<Option<usize>>` per call.
        let mut fpha_local_index: Vec<Option<usize>> = vec![None; ctx.n_hydros];
        for (local_idx, &h_idx) in fpha_hydro_indices.iter().enumerate() {
            fpha_local_index[h_idx] = Some(local_idx);
        }

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
        // the post-equipment column cursor (`post_equipment_col_start`), which
        // single-owns the shared start of the empty withdrawal/operational/NCS
        // regions so a future family inserted before NCS cannot leave this
        // `n_hydros == 0` start on a stale cursor.
        let active_ncs_indices = identify_active_ncs(ctx, stage);
        let n_active_ncs = active_ncs_indices.len();
        let col_ncs_start = if ctx.n_hydros > 0 {
            idx.generation_below_slack.end
        } else {
            idx.post_equipment_col_start()
        };
        let col_ncs_end = col_ncs_start + n_active_ncs * n_blks;

        // Row offsets: z_inflow, water balance, load balance, FPHA, evap, operational, generic.
        // z_inflow starts at row 0; state pinning is applied via column bounds.
        // The dual-relevant structural prefix of view.dual is empty because state
        // pinning uses column bounds. Cut-subgradient extraction reads
        // view.reduced_costs; n_dual_relevant on the row side is unused by the cut path.
        let n_dual_relevant = 0_usize;
        let n_op_rows = ctx.n_hydros * n_blks;
        // First operational-violation row: non-empty whenever `hydro_count > 0`
        // and then the canonical `min_generation_rows.start`; with no hydros the
        // four operational-violation row blocks normalise to `0..0`, so fall back
        // to the post-equipment row cursor (`post_equipment_row_start`), the
        // single-owned start every empty operational-violation row block shares.
        // Reproduces the `row_min_generation_start()` accessor's selection at
        // construction time (no `self` yet) so the fishing-row start that follows
        // is byte-identical. Sharing that one cursor is what keeps the empty-hydro
        // row layout from drifting onto a stale `> 0`-branch offset when a future
        // row family is inserted before these blocks.
        let row_min_generation_start = if ctx.n_hydros > 0 {
            idx.min_generation_rows.start
        } else {
            idx.post_equipment_row_start()
        };

        // One fishing row per anticipated plant at every stage (always-active).
        let n_anticipated_fishing_rows = ctx.n_anticipated;
        let row_anticipated_fishing_start = row_min_generation_start + n_op_rows;

        // Anticipated-state-out definition rows: one per ACTIVE plant. Active is
        // the single-owner gate `StageIndexer::is_anticipated_decision_active`
        // (the per-plant counterpart of `anticipated_decision_active_at_stage`);
        // inactive plants emit no row. Placed immediately after fishing rows,
        // before generic rows.
        let n_stages = ctx.resolved.bounds.n_stages();
        let n_anticipated_state_out_def_rows = (0..ctx.n_anticipated)
            .filter(|&local_idx| idx.is_anticipated_decision_active(local_idx, stage_idx, n_stages))
            .count();
        let row_anticipated_state_out_def_start =
            row_anticipated_fishing_start + n_anticipated_fishing_rows;
        let row_generic_start =
            row_anticipated_state_out_def_start + n_anticipated_state_out_def_rows;

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

        let num_cols = col_generic_slack_start + generic.n_generic_slack_cols;
        let num_rows = row_generic_start + generic.n_generic_rows;
        let zeta = stage.blocks.iter().map(|b| b.duration_hours).sum::<f64>() * M3S_TO_HM3;

        // The anticipated blocks normalise to `0..0` when empty; the decision and
        // state-out column starts are the thermal-block end plus the (possibly
        // zero) anticipated count.
        let anticipated = AnticipatedLayout {
            col_anticipated_decision_start: idx.thermal.end,
            col_anticipated_state_out_start: idx.thermal.end + ctx.n_anticipated,
            row_anticipated_state_out_def_start,
            n_anticipated_state_out_def_rows,
            row_anticipated_fishing_start,
            n_anticipated_fishing_rows,
        };

        Self {
            n_blks,
            n_h: ctx.n_hydros,
            lag_order: ctx.max_par_order,
            n_anticipated: ctx.n_anticipated,
            k_max: ctx.k_max,
            n_ant_state,
            anticipated,
            col_ncs_start,
            n_ncs: n_active_ncs,
            active_ncs_indices,
            col_pumping_start,
            n_pumping,
            num_cols,
            row_generic_start,
            num_rows,
            n_generic_rows: generic.n_generic_rows,
            n_dual_relevant,
            zeta,
            fpha_hydro_indices,
            fpha_local_index,
            fpha_planes_per_hydro,
            evap_hydro_indices,
            generic_constraint_rows: generic.generic_constraint_rows,
            indexer: idx,
        }
    }

    /// Resolve a block-major LP column address: `start + entity * n_blks + blk`.
    ///
    /// The block-major contract is fixed: the per-entity column count is
    /// `n_blks`, the entity index is the OUTER (stride) factor and the block is
    /// the INNER offset. The transposed form `blk * n_entities + entity` is the
    /// wrong-but-compiling alternative — it produces the same length region but
    /// interleaves columns across entities, so a coefficient lands on the wrong
    /// (entity, block) and the LP is silently misbuilt. The stride arithmetic
    /// lives one level down in [`BlockGrid::flat`](crate::indexer::BlockGrid::flat),
    /// the single owner; this method delegates there, and every flat-shape
    /// per-family accessor below delegates here (the deficit accessor uses the
    /// 3-term [`BlockGrid::deficit`](crate::indexer::BlockGrid::deficit) instead).
    #[inline]
    pub(crate) fn block_col(&self, start: usize, entity: usize, blk: usize) -> usize {
        self.indexer.block_grid().flat(start, entity, blk)
    }

    /// Turbine-flow column for hydro `h_idx`, block `blk`.
    #[inline]
    pub(crate) fn turbine_col(&self, h_idx: usize, blk: usize) -> usize {
        self.block_col(self.col_turbine_start(), h_idx, blk)
    }

    /// Spillage column for hydro `h_idx`, block `blk`.
    #[inline]
    pub(crate) fn spillage_col(&self, h_idx: usize, blk: usize) -> usize {
        self.block_col(self.col_spillage_start(), h_idx, blk)
    }

    /// Diversion-flow column for hydro `h_idx`, block `blk`.
    #[inline]
    pub(crate) fn diversion_col(&self, h_idx: usize, blk: usize) -> usize {
        self.block_col(self.col_diversion_start(), h_idx, blk)
    }

    /// FPHA generation column for FPHA-local index `local_idx`, block `blk`.
    #[inline]
    pub(crate) fn generation_col(&self, local_idx: usize, blk: usize) -> usize {
        self.block_col(self.col_generation_start(), local_idx, blk)
    }

    /// Forward line-flow column for line `l_idx`, block `blk`.
    #[inline]
    pub(crate) fn line_fwd_col(&self, l_idx: usize, blk: usize) -> usize {
        self.block_col(self.col_line_fwd_start(), l_idx, blk)
    }

    /// Reverse line-flow column for line `l_idx`, block `blk`.
    #[inline]
    pub(crate) fn line_rev_col(&self, l_idx: usize, blk: usize) -> usize {
        self.block_col(self.col_line_rev_start(), l_idx, blk)
    }

    /// Outflow-below-minimum slack column for hydro `h_idx`, block `blk`.
    #[inline]
    pub(crate) fn outflow_below_col(&self, h_idx: usize, blk: usize) -> usize {
        self.block_col(self.col_outflow_below_start(), h_idx, blk)
    }

    /// Outflow-above-maximum slack column for hydro `h_idx`, block `blk`.
    #[inline]
    pub(crate) fn outflow_above_col(&self, h_idx: usize, blk: usize) -> usize {
        self.block_col(self.col_outflow_above_start(), h_idx, blk)
    }

    /// Turbine-below-minimum slack column for hydro `h_idx`, block `blk`.
    #[inline]
    pub(crate) fn turbine_below_col(&self, h_idx: usize, blk: usize) -> usize {
        self.block_col(self.col_turbine_below_start(), h_idx, blk)
    }

    /// Generation-below-minimum slack column for hydro `h_idx`, block `blk`.
    #[inline]
    pub(crate) fn generation_below_col(&self, h_idx: usize, blk: usize) -> usize {
        self.block_col(self.col_generation_below_start(), h_idx, blk)
    }

    /// Evaporation-outflow column for evaporation-local index `local_idx`.
    ///
    /// Reserves [`EVAP_COLS_PER_HYDRO`] columns per evaporating hydro; this
    /// accessor returns the [`EVAP_FLOW_OFFSET`] column. Stage-level (no block).
    #[inline]
    pub(crate) fn evap_flow_col(&self, local_idx: usize) -> usize {
        self.col_evap_start() + local_idx * EVAP_COLS_PER_HYDRO + EVAP_FLOW_OFFSET
    }

    /// `f_evap_plus` (under-evaporation slack) column for evaporation-local
    /// index `local_idx` (the [`EVAP_F_PLUS_OFFSET`] column). Stage-level.
    #[inline]
    pub(crate) fn evap_f_plus_col(&self, local_idx: usize) -> usize {
        self.col_evap_start() + local_idx * EVAP_COLS_PER_HYDRO + EVAP_F_PLUS_OFFSET
    }

    /// `f_evap_minus` (over-evaporation slack) column for evaporation-local
    /// index `local_idx` (the [`EVAP_F_MINUS_OFFSET`] column). Stage-level.
    #[inline]
    pub(crate) fn evap_f_minus_col(&self, local_idx: usize) -> usize {
        self.col_evap_start() + local_idx * EVAP_COLS_PER_HYDRO + EVAP_F_MINUS_OFFSET
    }

    /// Deficit column for bus `b_idx`, segment `seg_idx`, block `blk`.
    ///
    /// Three-term stride: the deficit region uses a uniform per-bus span of
    /// `max_deficit_segments * n_blks` columns, then `n_blks` per segment, then
    /// the block. Returns
    /// `col_deficit_start + b_idx * max_deficit_segments * n_blks + seg_idx * n_blks + blk`.
    #[inline]
    pub(crate) fn deficit_col(&self, b_idx: usize, seg_idx: usize, blk: usize) -> usize {
        self.col_deficit_start()
            + b_idx * self.max_deficit_segments() * self.n_blks
            + seg_idx * self.n_blks
            + blk
    }

    // ── Indexer read-through accessors ──────────────────────────────────────────
    // Each delegates to the embedded indexer's owning range/scalar, so the offset
    // lives in one place (`self.indexer`) and cannot drift from a flattened copy.

    /// Theta (future-cost) column; delegates to `self.indexer.theta`.
    #[inline]
    #[must_use]
    pub(crate) fn col_theta(&self) -> usize {
        self.indexer.theta
    }

    /// First incoming-storage column; delegates to `self.indexer.storage_in.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_storage_in_start(&self) -> usize {
        self.indexer.storage_in.start
    }

    /// First AR-lag column; delegates to `self.indexer.inflow_lags.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_inflow_lags_start(&self) -> usize {
        self.indexer.inflow_lags.start
    }

    /// First turbine-flow column; delegates to `self.indexer.turbine.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_turbine_start(&self) -> usize {
        self.indexer.turbine.start
    }

    /// First spillage column; delegates to `self.indexer.spillage.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_spillage_start(&self) -> usize {
        self.indexer.spillage.start
    }

    /// First diversion-flow column; delegates to `self.indexer.diversion.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_diversion_start(&self) -> usize {
        self.indexer.diversion.start
    }

    /// First thermal column; delegates to `self.indexer.thermal.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_thermal_start(&self) -> usize {
        self.indexer.thermal.start
    }

    /// First forward line-flow column; delegates to `self.indexer.line_fwd.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_line_fwd_start(&self) -> usize {
        self.indexer.line_fwd.start
    }

    /// First reverse line-flow column; delegates to `self.indexer.line_rev.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_line_rev_start(&self) -> usize {
        self.indexer.line_rev.start
    }

    /// First deficit column; delegates to `self.indexer.deficit.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_deficit_start(&self) -> usize {
        self.indexer.deficit.start
    }

    /// Maximum deficit segments across buses; delegates to
    /// `self.indexer.max_deficit_segments`.
    #[inline]
    #[must_use]
    pub(crate) fn max_deficit_segments(&self) -> usize {
        self.indexer.max_deficit_segments
    }

    /// First excess column; delegates to `self.indexer.excess.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_excess_start(&self) -> usize {
        self.indexer.excess.start
    }

    /// First inflow-slack column; the `inflow_slack` block normalises to `0..0`
    /// without the penalty, so this is the excess-block end cursor — delegates to
    /// `self.indexer.excess.end`.
    #[inline]
    #[must_use]
    pub(crate) fn col_inflow_slack_start(&self) -> usize {
        self.indexer.excess.end
    }

    /// First z-inflow column; delegates to `self.indexer.z_inflow.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_z_inflow_start(&self) -> usize {
        self.indexer.z_inflow.start
    }

    /// First z-inflow definition row; delegates to
    /// `self.indexer.z_inflow_row_start`.
    #[inline]
    #[must_use]
    pub(crate) fn row_z_inflow_start(&self) -> usize {
        self.indexer.z_inflow_row_start
    }

    /// First anticipated-state column; delegates to
    /// `self.indexer.anticipated_state.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_anticipated_state_start(&self) -> usize {
        self.indexer.anticipated_state.start
    }

    /// First water-balance row; delegates to `self.indexer.water_balance.start`.
    #[inline]
    #[must_use]
    pub(crate) fn row_water_balance_start(&self) -> usize {
        self.indexer.water_balance.start
    }

    /// First load-balance row; delegates to `self.indexer.load_balance.start`.
    #[inline]
    #[must_use]
    pub(crate) fn row_load_balance_start(&self) -> usize {
        self.indexer.load_balance.start
    }

    /// First FPHA row; the FPHA block follows the load-balance rows, so this is
    /// the load-balance end cursor — delegates to `self.indexer.load_balance.end`.
    #[inline]
    #[must_use]
    pub(crate) fn row_fpha_start(&self) -> usize {
        self.indexer.load_balance.end
    }

    /// Column-side state dimension; delegates to `self.indexer.n_state`.
    #[inline]
    #[must_use]
    pub(crate) fn n_state(&self) -> usize {
        self.indexer.n_state
    }

    // ── Empty-block-cursor read-through accessors ────────────────────────────────
    // Each delegates to an indexer empty-block-cursor accessor or reproduces the
    // constructor's `n_hydros > 0` selection, so the offset lives in one place
    // (`self.indexer`) and cannot drift from a flattened copy. These differ from
    // the plain read-throughs above: the underlying public range normalises to
    // `0..0` when its family is empty, so a bare `self.indexer.<range>.start` would
    // return `0` rather than the real cursor — silently misbuilding the empty-hydro
    // layout. Each accessor returns the empty-block cursor instead.

    /// Start of FPHA generation columns (one per FPHA hydro per block); delegates
    /// to `self.indexer.generation_col_start()`.
    ///
    /// Layout within this region: `col_generation_start() + local_fpha_idx * n_blks + blk`.
    /// The public `generation` range normalises to `0..0` with no FPHA hydros, so
    /// this reads the indexer's empty-block cursor — never `0` for a real LP.
    #[inline]
    #[must_use]
    pub(crate) fn col_generation_start(&self) -> usize {
        self.indexer.generation_col_start()
    }

    /// Start of evaporation columns (after FPHA generation columns); delegates to
    /// `self.indexer.evap_col_start()`.
    ///
    /// [`EVAP_COLS_PER_HYDRO`] stage-level columns per evaporation hydro
    /// (evaporation outflow, `f_evap_plus`, `f_evap_minus`). Address the three via
    /// the `evap_flow_col` / `evap_f_plus_col` / `evap_f_minus_col` accessors rather
    /// than open-coding the stride. The indexer's `evap_indices` list is empty with
    /// no evaporation hydros, so this reads the empty-block cursor — never `0` for a
    /// real LP.
    #[inline]
    #[must_use]
    pub(crate) fn col_evap_start(&self) -> usize {
        self.indexer.evap_col_start()
    }

    /// Start of evaporation constraint rows (after FPHA rows); delegates to
    /// `self.indexer.fpha_rows_end()`.
    ///
    /// One equality row per evaporation hydro. Layout: `row_evap_start() + local_evap_idx`.
    /// The FPHA and evaporation row blocks both normalise to empty, so this reads the
    /// indexer's `fpha_rows_end()` cursor rather than the `0..0` public range — never
    /// `0` for a real LP.
    #[inline]
    #[must_use]
    pub(crate) fn row_evap_start(&self) -> usize {
        self.indexer.fpha_rows_end()
    }

    /// Start of under-withdrawal slack columns (after evaporation columns).
    ///
    /// One stage-level column per operating hydro. Layout: `col_withdrawal_neg_start() + h`.
    /// Branches on `self.n_h > 0` — the constructor's `ctx.n_hydros > 0` discriminant.
    /// With no hydros the `withdrawal_slack_neg` range normalises to `0..0`, so the
    /// fallback is the post-equipment column cursor `post_equipment_col_start()`, NOT
    /// the `0` a bare `withdrawal_slack_neg.start` would return.
    #[inline]
    #[must_use]
    pub(crate) fn col_withdrawal_neg_start(&self) -> usize {
        if self.n_h > 0 {
            self.indexer.withdrawal_slack_neg.start
        } else {
            self.indexer.post_equipment_col_start()
        }
    }

    /// Start of over-withdrawal slack columns (after under-withdrawal slacks).
    ///
    /// One stage-level column per operating hydro. Layout: `col_withdrawal_pos_start() + h`.
    /// Branches on `self.n_h > 0`; the no-hydro fallback is `post_equipment_col_start()`,
    /// NOT `0` (see [`Self::col_withdrawal_neg_start`]).
    #[inline]
    #[must_use]
    pub(crate) fn col_withdrawal_pos_start(&self) -> usize {
        if self.n_h > 0 {
            self.indexer.withdrawal_slack_pos.start
        } else {
            self.indexer.post_equipment_col_start()
        }
    }

    /// Start of outflow-below-minimum slack columns (one per hydro per block).
    ///
    /// Inserted after withdrawal slack columns.
    /// Layout: `col_outflow_below_start() + h_idx * n_blks + blk`.
    /// Branches on `self.n_h > 0`; the no-hydro fallback is `post_equipment_col_start()`,
    /// NOT `0` (see [`Self::col_withdrawal_neg_start`]).
    #[inline]
    #[must_use]
    pub(crate) fn col_outflow_below_start(&self) -> usize {
        if self.n_h > 0 {
            self.indexer.outflow_below_slack.start
        } else {
            self.indexer.post_equipment_col_start()
        }
    }

    /// Start of outflow-above-maximum slack columns (one per hydro per block).
    ///
    /// Layout: `col_outflow_above_start() + h_idx * n_blks + blk`.
    /// Branches on `self.n_h > 0`; the no-hydro fallback is `post_equipment_col_start()`,
    /// NOT `0` (see [`Self::col_withdrawal_neg_start`]).
    #[inline]
    #[must_use]
    pub(crate) fn col_outflow_above_start(&self) -> usize {
        if self.n_h > 0 {
            self.indexer.outflow_above_slack.start
        } else {
            self.indexer.post_equipment_col_start()
        }
    }

    /// Start of turbine-below-minimum slack columns (one per hydro per block).
    ///
    /// Layout: `col_turbine_below_start() + h_idx * n_blks + blk`.
    /// Branches on `self.n_h > 0`; the no-hydro fallback is `post_equipment_col_start()`,
    /// NOT `0` (see [`Self::col_withdrawal_neg_start`]).
    #[inline]
    #[must_use]
    pub(crate) fn col_turbine_below_start(&self) -> usize {
        if self.n_h > 0 {
            self.indexer.turbine_below_slack.start
        } else {
            self.indexer.post_equipment_col_start()
        }
    }

    /// Start of generation-below-minimum slack columns (one per hydro per block).
    ///
    /// Layout: `col_generation_below_start() + h_idx * n_blks + blk`.
    /// Branches on `self.n_h > 0`; the no-hydro fallback is `post_equipment_col_start()`,
    /// NOT `0` (see [`Self::col_withdrawal_neg_start`]).
    #[inline]
    #[must_use]
    pub(crate) fn col_generation_below_start(&self) -> usize {
        if self.n_h > 0 {
            self.indexer.generation_below_slack.start
        } else {
            self.indexer.post_equipment_col_start()
        }
    }

    /// Start of minimum-outflow constraint rows (one per hydro per block, after evaporation rows).
    ///
    /// Layout: `row_min_outflow_start() + h_idx * n_blks + blk`.
    /// Branches on `self.n_h > 0` — the constructor's `ctx.n_hydros > 0` discriminant.
    /// With no hydros the operational-violation row blocks normalise to `0..0`, so the
    /// fallback is the post-equipment row cursor `post_equipment_row_start()`, NOT the
    /// `0` a bare `min_outflow_rows.start` would return.
    #[inline]
    #[must_use]
    pub(crate) fn row_min_outflow_start(&self) -> usize {
        if self.n_h > 0 {
            self.indexer.min_outflow_rows.start
        } else {
            self.indexer.post_equipment_row_start()
        }
    }

    /// Start of maximum-outflow constraint rows (one per hydro per block).
    ///
    /// Layout: `row_max_outflow_start() + h_idx * n_blks + blk`.
    /// Branches on `self.n_h > 0`; the no-hydro fallback is `post_equipment_row_start()`,
    /// NOT `0` (see [`Self::row_min_outflow_start`]).
    #[inline]
    #[must_use]
    pub(crate) fn row_max_outflow_start(&self) -> usize {
        if self.n_h > 0 {
            self.indexer.max_outflow_rows.start
        } else {
            self.indexer.post_equipment_row_start()
        }
    }

    /// Start of minimum-turbine constraint rows (one per hydro per block).
    ///
    /// Layout: `row_min_turbine_start() + h_idx * n_blks + blk`.
    /// Branches on `self.n_h > 0`; the no-hydro fallback is `post_equipment_row_start()`,
    /// NOT `0` (see [`Self::row_min_outflow_start`]).
    #[inline]
    #[must_use]
    pub(crate) fn row_min_turbine_start(&self) -> usize {
        if self.n_h > 0 {
            self.indexer.min_turbine_rows.start
        } else {
            self.indexer.post_equipment_row_start()
        }
    }

    /// Start of minimum-generation constraint rows (one per hydro per block).
    ///
    /// Layout: `row_min_generation_start() + h_idx * n_blks + blk`.
    /// Branches on `self.n_h > 0`; the no-hydro fallback is `post_equipment_row_start()`,
    /// NOT `0` (see [`Self::row_min_outflow_start`]).
    #[inline]
    #[must_use]
    pub(crate) fn row_min_generation_start(&self) -> usize {
        if self.n_h > 0 {
            self.indexer.min_generation_rows.start
        } else {
            self.indexer.post_equipment_row_start()
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
    use std::collections::{BTreeMap, HashMap};

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

    use super::{
        EVAP_COLS_PER_HYDRO, EVAP_F_MINUS_OFFSET, EVAP_F_PLUS_OFFSET, EVAP_FLOW_OFFSET,
        ResolvedTables, StageLayout, TemplateBuildCtx,
    };

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
            layout.col_turbine_start(),
            idx.theta + 1,
            "col_turbine_start must equal idx.theta + 1 with zero anticipated"
        );
    }

    // ── FPHA-local inverse map ───────────────────────────────────────────────

    /// Owns the data needed to construct a three-hydro `TemplateBuildCtx` with a
    /// single FPHA hydro at system index 1 (the other two use constant
    /// productivity), so `StageLayout::new` derives `fpha_hydro_indices == [1]`.
    struct FphaMixFixtures {
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

    impl FphaMixFixtures {
        fn new() -> Self {
            use crate::hydro_models::{EvaporationModel, FphaPlane, ResolvedProductionModel};

            let constant = ResolvedProductionModel::ConstantProductivity { productivity: 0.0 };
            let fpha = ResolvedProductionModel::Fpha {
                planes: vec![FphaPlane {
                    intercept: 0.0,
                    gamma_v: 0.0,
                    gamma_q: 0.0,
                    gamma_s: 0.0,
                }],
            };
            // models[hydro][stage]: hydro 1 is FPHA, hydros 0 and 2 are constant.
            let models = vec![vec![constant.clone()], vec![fpha], vec![constant]];
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
                production_models: ProductionModelSet::new(models, 3, 1),
                evaporation_models: EvaporationModelSet::new(vec![
                    EvaporationModel::None,
                    EvaporationModel::None,
                    EvaporationModel::None,
                ]),
            }
        }

        fn make_ctx(&self) -> TemplateBuildCtx<'_> {
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
                diversion_upstream: HashMap::new(),
                n_hydros: 3,
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

    /// `StageLayout::new` inverts `fpha_hydro_indices` into `fpha_local_index`:
    /// the FPHA hydro at system index 1 of three maps to local index 0, and the
    /// two non-FPHA hydros stay `None`, giving `[None, Some(0), None]`.
    #[test]
    fn stage_layout_populates_fpha_local_index_inverse_map() {
        let fixtures = FphaMixFixtures::new();
        let ctx = fixtures.make_ctx();
        let stage = minimal_stage();
        let layout = StageLayout::new(&ctx, &stage, 0);

        assert_eq!(
            layout.fpha_hydro_indices,
            vec![1],
            "only the system-index-1 hydro uses FPHA"
        );
        assert_eq!(
            layout.fpha_local_index,
            vec![None, Some(0), None],
            "fpha_local_index inverts fpha_hydro_indices over n_h = 3"
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
            layout.anticipated.col_anticipated_decision_start,
            layout.col_thermal_start(),
            "col_anticipated_decision_start must equal col_thermal_start \
             when n_thermals=0 (no thermal per-block cols)"
        );
        assert_eq!(
            layout.anticipated.col_anticipated_state_out_start,
            layout.anticipated.col_anticipated_decision_start + n_anticipated,
            "col_anticipated_state_out_start == col_anticipated_decision_start + n_anticipated"
        );
        assert_eq!(
            layout.col_line_fwd_start(),
            layout.anticipated.col_anticipated_state_out_start + n_anticipated,
            "col_line_fwd_start == col_anticipated_state_out_start + n_anticipated"
        );
        // Verify the separation between thermal_start and line_fwd_start is exactly 2*n_anticipated
        // (n_anticipated cols for anticipated_decision + n_anticipated cols for anticipated_state_out).
        assert_eq!(
            layout.col_line_fwd_start() - layout.col_thermal_start(),
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
            layout.col_turbine_start(),
            expected_col_turbine_start,
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
            layout.anticipated.row_anticipated_fishing_start,
            layout.row_min_generation_start() + n_op_rows,
            "row_anticipated_fishing_start must equal row_min_generation_start + n_op_rows"
        );
        // Always-active: both plants active at every stage → 2 fishing rows.
        assert_eq!(
            layout.anticipated.n_anticipated_fishing_rows, 2,
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
                layout.anticipated.n_anticipated_fishing_rows, expected,
                "n_anticipated_fishing_rows must equal {expected} at stage_idx={stage_idx}"
            );
        }
    }

    /// `num_rows` does not include state-fixing rows; the LP row layout starts
    /// directly with `z_inflow_rows` at row 0.
    ///
    /// State pinning uses column bounds, so there is no `[0, n_state)` row
    /// prefix. `num_rows` equals the count of structural rows only (`z_inflow`,
    /// water balance, load balance, FPHA, evap, operational, fishing,
    /// `anticipated_state_out_def`, generic).
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

        // num_rows for this zero-hydro fixture: only the anticipated_fishing
        // block contributes (2 active plants at stage 0). All other row blocks
        // are 0 (no hydros, no buses, no FPHA, no evap).
        let observed = layout.num_rows;
        assert_eq!(
            observed, 2,
            "num_rows equals anticipated_fishing_rows (2) for this fixture"
        );

        // Reference value: if state-fixing rows existed, num_rows would be observed + n_state.
        let num_rows_if_state_rows_existed = observed + n_state;
        assert_eq!(
            num_rows_if_state_rows_existed, 8,
            "observed + n_state is 8 for this fixture"
        );
        // Structural invariant proving the reduction: row_water_balance_start
        // equals ctx.n_hydros (no n_state offset). With state-fixing rows it
        // would be n_state + ctx.n_hydros.
        assert_eq!(
            layout.row_water_balance_start(),
            ctx.n_hydros,
            "row_water_balance_start does not include the n_state offset"
        );
    }

    // ── Anticipated-decision range tests ──────────────────────────────────────

    /// Build a `ResolvedBounds` with zero entities but the given `n_stages`.
    ///
    /// Used to exercise the `StageIndexer::is_anticipated_decision_active` gate
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
            layout.anticipated.col_anticipated_state_out_start,
            layout.anticipated.col_anticipated_decision_start + 2,
            "state_out columns must be immediately after anticipated_decision"
        );
        assert_eq!(
            layout.col_line_fwd_start(),
            layout.anticipated.col_anticipated_state_out_start + 2,
            "line_fwd must be immediately after state_out columns"
        );
        assert_eq!(layout.anticipated.n_anticipated_state_out_def_rows, 2);
        assert_eq!(
            layout.anticipated.row_anticipated_state_out_def_start,
            layout.anticipated.row_anticipated_fishing_start
                + layout.anticipated.n_anticipated_fishing_rows
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

        assert_eq!(layout.anticipated.n_anticipated_state_out_def_rows, 0);
        // Column block stays allocated regardless of activity.
        assert_eq!(
            layout.anticipated.col_anticipated_state_out_start,
            layout.anticipated.col_anticipated_decision_start + 2
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
            layout.anticipated.col_anticipated_state_out_start,
            layout.anticipated.col_anticipated_decision_start,
            "col_anticipated_state_out_start must equal col_anticipated_decision_start when n_anticipated=0"
        );
        assert_eq!(layout.anticipated.n_anticipated_state_out_def_rows, 0);
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
                // This fixture probes only the column-reservation arithmetic, which
                // consumes `n_pumping` (not the entity slice). The station slice is
                // left empty while `n_pumping` is sourced from the pumping-aware
                // `ResolvedBounds` so `StageLayout::new` reserves the right block;
                // the slice/`pumping_pos` threading is covered by the
                // `build_template_build_ctx` tests in `template.rs`.
                pumping_stations: &[],
                pumping_pos: BTreeMap::new(),
                n_pumping: self.bounds.n_pumping(),
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
        assert_eq!(
            ctx.resolved.bounds.n_pumping(),
            0,
            "fixture has no pumping stations"
        );
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
        assert_eq!(layout.col_turbine_start(), expected_start);
        assert_eq!(layout.col_thermal_start(), expected_start);
        assert_eq!(layout.col_line_fwd_start(), expected_start);
        assert_eq!(layout.col_deficit_start(), expected_start);
        assert_eq!(layout.col_excess_start(), expected_start);
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
            ctx.resolved.bounds.n_pumping(),
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

    // ── block-major column-accessor arithmetic equivalence ─────────────────────

    /// Every `#[inline]` column accessor returns the exact `usize` its open-coded
    /// formula returned. Built over a `n_blks = 4` layout and probed with
    /// `entity >= 1`, `blk >= 1`, and `seg >= 1` so a transposed stride
    /// (`blk * n_entities + entity`) or a swapped evap offset would differ from
    /// the open-coded expression and fail the assertion.
    #[test]
    fn column_accessors_match_open_coded_formulas() {
        // Multi-block, zero-entity layout: the block-major `col_*_start` fields
        // and `n_blks` are populated; the accessor reads the same fields the
        // open-coded formula reads, so this pins each accessor's arithmetic.
        let fixtures = ZeroEntityFixtures::new();
        let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
        let stage = PumpingFixtures::stage_with_blocks(4);
        let layout = StageLayout::new(&ctx, &stage, 0);
        let n_blks = layout.n_blks;
        assert_eq!(n_blks, 4, "fixture must build a 4-block layout");

        // Generic block_col against its definition, across a grid that makes the
        // entity (outer) and block (inner) factors distinguishable.
        for entity in [0_usize, 1, 2, 5] {
            for blk in 0..n_blks {
                assert_eq!(
                    layout.block_col(layout.col_turbine_start(), entity, blk),
                    layout.col_turbine_start() + entity * n_blks + blk,
                    "block_col(entity={entity}, blk={blk})"
                );
            }
        }

        // Per-family block-major accessors, each owning its `col_*_start`.
        for entity in [0_usize, 1, 3] {
            for blk in 0..n_blks {
                assert_eq!(
                    layout.turbine_col(entity, blk),
                    layout.col_turbine_start() + entity * n_blks + blk,
                    "turbine_col"
                );
                assert_eq!(
                    layout.spillage_col(entity, blk),
                    layout.col_spillage_start() + entity * n_blks + blk,
                    "spillage_col"
                );
                assert_eq!(
                    layout.diversion_col(entity, blk),
                    layout.col_diversion_start() + entity * n_blks + blk,
                    "diversion_col"
                );
                assert_eq!(
                    layout.generation_col(entity, blk),
                    layout.col_generation_start() + entity * n_blks + blk,
                    "generation_col"
                );
                assert_eq!(
                    layout.line_fwd_col(entity, blk),
                    layout.col_line_fwd_start() + entity * n_blks + blk,
                    "line_fwd_col"
                );
                assert_eq!(
                    layout.line_rev_col(entity, blk),
                    layout.col_line_rev_start() + entity * n_blks + blk,
                    "line_rev_col"
                );
                assert_eq!(
                    layout.outflow_below_col(entity, blk),
                    layout.col_outflow_below_start() + entity * n_blks + blk,
                    "outflow_below_col"
                );
                assert_eq!(
                    layout.outflow_above_col(entity, blk),
                    layout.col_outflow_above_start() + entity * n_blks + blk,
                    "outflow_above_col"
                );
                assert_eq!(
                    layout.turbine_below_col(entity, blk),
                    layout.col_turbine_below_start() + entity * n_blks + blk,
                    "turbine_below_col"
                );
                assert_eq!(
                    layout.generation_below_col(entity, blk),
                    layout.col_generation_below_start() + entity * n_blks + blk,
                    "generation_below_col"
                );
            }
        }

        // Evaporation accessors: stage-level, EVAP_COLS_PER_HYDRO-strided. The
        // three within-hydro offsets must map flow→0, f_plus→1, f_minus→2.
        for local_idx in [0_usize, 1, 4] {
            assert_eq!(
                layout.evap_flow_col(local_idx),
                layout.col_evap_start() + local_idx * EVAP_COLS_PER_HYDRO + EVAP_FLOW_OFFSET,
                "evap_flow_col"
            );
            assert_eq!(
                layout.evap_f_plus_col(local_idx),
                layout.col_evap_start() + local_idx * EVAP_COLS_PER_HYDRO + EVAP_F_PLUS_OFFSET,
                "evap_f_plus_col"
            );
            assert_eq!(
                layout.evap_f_minus_col(local_idx),
                layout.col_evap_start() + local_idx * EVAP_COLS_PER_HYDRO + EVAP_F_MINUS_OFFSET,
                "evap_f_minus_col"
            );
            // The three columns are consecutive and ordered flow < plus < minus.
            assert_eq!(
                layout.evap_f_plus_col(local_idx),
                layout.evap_flow_col(local_idx) + 1
            );
            assert_eq!(
                layout.evap_f_minus_col(local_idx),
                layout.evap_flow_col(local_idx) + 2
            );
        }
        // The evap offset consts are exactly 0/1/2 in order.
        assert_eq!(EVAP_FLOW_OFFSET, 0);
        assert_eq!(EVAP_F_PLUS_OFFSET, 1);
        assert_eq!(EVAP_F_MINUS_OFFSET, 2);
        assert_eq!(EVAP_COLS_PER_HYDRO, 3);

        // Deficit three-term stride: per-bus span, per-segment span, then block.
        for b_idx in [0_usize, 1, 2] {
            for seg_idx in [0_usize, 1] {
                for blk in 0..n_blks {
                    assert_eq!(
                        layout.deficit_col(b_idx, seg_idx, blk),
                        layout.col_deficit_start()
                            + b_idx * layout.max_deficit_segments() * n_blks
                            + seg_idx * n_blks
                            + blk,
                        "deficit_col(b={b_idx}, seg={seg_idx}, blk={blk})"
                    );
                }
            }
        }
    }

    // ── post-equipment column cursor (no-hydro fork fallback) ───────────────────

    /// With `n_hydros == 0` every withdrawal / operational-violation / NCS column
    /// region is empty, so all their starts collapse onto the single
    /// post-equipment column cursor `col_evap_start`. A multi-block stage keeps
    /// that cursor non-trivial (not the degenerate one-column case), so a stale
    /// `> 0`-branch cursor leaking into the empty-hydro fallback would shift these
    /// starts off `col_evap_start` and fail here.
    #[test]
    fn post_equipment_col_start_matches_evap_col_start_when_no_hydros() {
        let fixtures = ZeroEntityFixtures::new();
        let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
        let stage = PumpingFixtures::stage_with_blocks(4);
        let layout = StageLayout::new(&ctx, &stage, 0);

        assert_eq!(ctx.n_hydros, 0, "fixture must have zero hydros");
        assert_eq!(layout.n_blks, 4, "fixture must build a 4-block layout");

        let post_equipment = layout.col_evap_start();
        assert_eq!(
            layout.col_ncs_start, post_equipment,
            "col_ncs_start must collapse onto col_evap_start when n_hydros == 0"
        );
        assert_eq!(
            layout.col_withdrawal_neg_start(),
            post_equipment,
            "col_withdrawal_neg_start"
        );
        assert_eq!(
            layout.col_withdrawal_pos_start(),
            post_equipment,
            "col_withdrawal_pos_start"
        );
        assert_eq!(
            layout.col_outflow_below_start(),
            post_equipment,
            "col_outflow_below_start"
        );
        assert_eq!(
            layout.col_outflow_above_start(),
            post_equipment,
            "col_outflow_above_start"
        );
        assert_eq!(
            layout.col_turbine_below_start(),
            post_equipment,
            "col_turbine_below_start"
        );
        assert_eq!(
            layout.col_generation_below_start(),
            post_equipment,
            "col_generation_below_start"
        );
    }

    // ── post-equipment row cursor (no-hydro fork fallback) ──────────────────────

    /// With `n_hydros == 0` every operational-violation row block is empty, so all
    /// four row starts collapse onto the single post-equipment row cursor
    /// `fpha_rows_end() + n_evap_hydros`. A multi-block stage keeps that cursor
    /// non-trivial (not the degenerate one-row case), so a stale `> 0`-branch
    /// cursor leaking into the empty-hydro fallback would shift these starts off
    /// the shared post-equipment row cursor and fail here.
    #[test]
    fn post_equipment_row_start_matches_evap_rows_end_when_no_hydros() {
        let fixtures = ZeroEntityFixtures::new();
        let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
        let stage = PumpingFixtures::stage_with_blocks(4);
        let layout = StageLayout::new(&ctx, &stage, 0);

        assert_eq!(ctx.n_hydros, 0, "fixture must have zero hydros");
        assert_eq!(layout.n_blks, 4, "fixture must build a 4-block layout");

        let post_equipment = layout.indexer.fpha_rows_end() + layout.indexer.n_evap_hydros;
        assert_eq!(
            layout.row_min_outflow_start(),
            post_equipment,
            "row_min_outflow_start must collapse onto fpha_rows_end() + n_evap_hydros when n_hydros == 0"
        );
        assert_eq!(
            layout.row_max_outflow_start(),
            post_equipment,
            "row_max_outflow_start"
        );
        assert_eq!(
            layout.row_min_turbine_start(),
            post_equipment,
            "row_min_turbine_start"
        );
        assert_eq!(
            layout.row_min_generation_start(),
            post_equipment,
            "row_min_generation_start"
        );
    }

    // ── Group-2 accessors: hydro-free divergence guard ──────────────────────────

    /// With `n_hydros == 0`, every Group-2 accessor must return the post-equipment
    /// cursor — `post_equipment_col_start()` for the eight column accessors,
    /// `post_equipment_row_start()` for the five row accessors. A bare
    /// `self.indexer.<range>.start` returns the `0` of the normalised `0..0` empty
    /// range; the equality assertions here pin the accessor to the real cursor
    /// instead, which is the silent misbuild this split exists to prevent.
    ///
    /// The column cursor is additionally asserted `!= 0`: the theta and state columns
    /// always precede the equipment/slack region, so `post_equipment_col_start()` is
    /// provably positive and a spurious bare-`.start` `0` is directly detectable. The
    /// row cursor is NOT asserted `!= 0`: with zero hydros AND zero buses no rows
    /// precede the operational-violation block, so `post_equipment_row_start()` is
    /// legitimately `0` here (asserting `!= 0` would test a false invariant). The
    /// non-zero-row divergence is covered end-to-end by the D01 hydro-free parity
    /// case, whose load-balance rows make the row cursor positive.
    #[test]
    fn group2_accessors_return_post_equipment_cursor_when_no_hydros() {
        let fixtures = ZeroEntityFixtures::new();
        let ctx = fixtures.make_ctx(0, 0, vec![], vec![]);
        let stage = PumpingFixtures::stage_with_blocks(4);
        let layout = StageLayout::new(&ctx, &stage, 0);

        assert_eq!(ctx.n_hydros, 0, "fixture must have zero hydros");
        assert_eq!(layout.n_blks, 4, "fixture must build a 4-block layout");

        // Column cursor: the eight column accessors collapse onto
        // `post_equipment_col_start()` (== `col_evap_start()`) with no hydros, and
        // that cursor is provably positive (theta + state columns precede it).
        let post_col = layout.indexer.post_equipment_col_start();
        assert_ne!(post_col, 0, "post-equipment column cursor must not be 0");
        for (value, name) in [
            (layout.col_generation_start(), "col_generation_start"),
            (layout.col_evap_start(), "col_evap_start"),
            (
                layout.col_withdrawal_neg_start(),
                "col_withdrawal_neg_start",
            ),
            (
                layout.col_withdrawal_pos_start(),
                "col_withdrawal_pos_start",
            ),
            (layout.col_outflow_below_start(), "col_outflow_below_start"),
            (layout.col_outflow_above_start(), "col_outflow_above_start"),
            (layout.col_turbine_below_start(), "col_turbine_below_start"),
            (
                layout.col_generation_below_start(),
                "col_generation_below_start",
            ),
        ] {
            assert_eq!(
                value, post_col,
                "{name} must equal post_equipment_col_start() (not 0) when n_hydros == 0"
            );
        }

        // Row cursor: `row_evap_start()` is `fpha_rows_end()`, which equals
        // `post_equipment_row_start()` (= `fpha_rows_end() + n_evap_hydros`) when
        // `n_evap_hydros == 0`. The four operational-violation row accessors collapse
        // onto that same cursor. Each must equal it, never a bare `.start`.
        let post_row = layout.indexer.post_equipment_row_start();
        for (value, name) in [
            (layout.row_evap_start(), "row_evap_start"),
            (layout.row_min_outflow_start(), "row_min_outflow_start"),
            (layout.row_max_outflow_start(), "row_max_outflow_start"),
            (layout.row_min_turbine_start(), "row_min_turbine_start"),
            (
                layout.row_min_generation_start(),
                "row_min_generation_start",
            ),
        ] {
            assert_eq!(
                value, post_row,
                "{name} must equal post_equipment_row_start() when n_hydros == 0"
            );
        }
    }
}
