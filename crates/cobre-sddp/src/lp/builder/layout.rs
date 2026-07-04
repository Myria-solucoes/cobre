use std::collections::{BTreeMap, HashMap};
use std::ops::Range;

use cobre_core::{
    BlockMode, Bus, CascadeTopology, ConstraintSense, EnergyContract, EntityId, GenericConstraint,
    Hydro, Line, LoadModel, NonControllableSource, PumpingStation, ResolvedBounds,
    ResolvedExchangeFactors, ResolvedGenericConstraintBounds, ResolvedLoadFactors,
    ResolvedNcsBounds, ResolvedNcsFactors, ResolvedPenalties, Stage, Thermal,
};
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::hydro_models::{
    EvaporationModel, EvaporationModelSet, ProductionModelSet, ResolvedProductionModel,
};
use crate::indexer::{BlockGrid, EvaporationIndices, StateLayout};
use crate::lead_time::SpreadResolution;

use super::{
    EVAP_COLS_PER_HYDRO, EVAP_F_MINUS_OFFSET, EVAP_F_PLUS_OFFSET, EVAP_FLOW_OFFSET,
    GenericConstraintRowEntry, M3S_TO_HM3, Phase, filling_phase,
};

/// Pre-resolved bound, penalty, and factor tables shared across all stages.
pub(crate) struct ResolvedTables<'a> {
    /// Resolved per-stage entity bounds.
    pub(crate) bounds: &'a ResolvedBounds,
    /// Resolved per-stage penalties.
    pub(crate) penalties: &'a ResolvedPenalties,
    /// `(constraint_idx, stage_id)` → active bound entries.
    pub(crate) resolved_generic_bounds: &'a ResolvedGenericConstraintBounds,
    /// Per-block load scaling factors.
    pub(crate) resolved_load_factors: &'a ResolvedLoadFactors,
    /// Per-block exchange capacity factors.
    pub(crate) resolved_exchange_factors: &'a ResolvedExchangeFactors,
    /// Per-stage NCS available generation bounds.
    pub(crate) resolved_ncs_bounds: &'a ResolvedNcsBounds,
    /// Per-block NCS generation scaling factors.
    pub(crate) resolved_ncs_factors: &'a ResolvedNcsFactors,
    /// `(parameter_id, stage_idx)` → resolved `f64`, queried for a
    /// [`cobre_core::CoefficientRef::Parameter`] term.
    pub(crate) resolved_parameters: &'a crate::resolved_parameters::ResolvedParameters,
}

/// System-level context shared across all stages during template construction.
///
/// Constructed once in `build_stage_templates` and borrowed by
/// `build_single_stage_template` for each study stage.
pub(crate) struct TemplateBuildCtx<'a> {
    pub(crate) hydros: &'a [Hydro],
    pub(crate) thermals: &'a [Thermal],
    pub(crate) lines: &'a [Line],
    pub(crate) buses: &'a [Bus],
    pub(crate) load_models: &'a [LoadModel],
    pub(crate) cascade: &'a CascadeTopology,
    /// Pre-resolved bound, penalty, and factor tables.
    pub(crate) resolved: ResolvedTables<'a>,
    /// Entity-id → canonical slot index. `BTreeMap`, not `HashMap`: an accidental
    /// iterating fill then emits entries in canonical `EntityId` order, not
    /// nondeterministic `HashMap` order (declaration-order bit-determinism;
    /// `csc_byte_identical_under_permuted_multi_entity_order`).
    pub(crate) hydro_pos: BTreeMap<EntityId, usize>,
    /// Thermal id → slot. `BTreeMap` for determinism, see `hydro_pos`.
    pub(crate) thermal_pos: BTreeMap<EntityId, usize>,
    /// Line id → slot. `BTreeMap` for determinism, see `hydro_pos`.
    pub(crate) line_pos: BTreeMap<EntityId, usize>,
    /// Bus id → slot. `BTreeMap` for determinism, see `hydro_pos`.
    pub(crate) bus_pos: BTreeMap<EntityId, usize>,
    pub(crate) par_lp: &'a PrecomputedPar,
    /// Resolved production models for all (hydro, stage) pairs.
    pub(crate) production_models: &'a ProductionModelSet,
    /// Resolved evaporation models for all hydro plants.
    pub(crate) evaporation_models: &'a EvaporationModelSet,
    /// Generic constraint definitions (expression, sense, slack config).
    pub(crate) generic_constraints: &'a [GenericConstraint],
    /// Non-controllable source entities, id-sorted.
    pub(crate) non_controllable_sources: &'a [NonControllableSource],
    /// Pumping station entities, id-sorted (canonical slot order).
    pub(crate) pumping_stations: &'a [PumpingStation],
    /// Station id → slot into `pumping_stations`. `BTreeMap` for determinism, see
    /// `hydro_pos`.
    pub(crate) pumping_pos: BTreeMap<EntityId, usize>,
    /// Full station count, asserted `== bounds.n_pumping()` at construction. The
    /// dense per-stage column-block stride: every station keeps a column at every
    /// stage, a commissioning-dormant one zeroed to `[0, 0]` rather than omitted.
    pub(crate) n_pumping: usize,
    /// Energy contract entities, id-sorted (canonical slot order). One slice for
    /// both directions; the import/export split is derived at fill time from
    /// `contract_type`, not pre-partitioned.
    pub(crate) contracts: &'a [EnergyContract],
    /// Contract id → slot into `contracts`. `BTreeMap` for determinism, see
    /// `hydro_pos`.
    pub(crate) contract_pos: BTreeMap<EntityId, usize>,
    /// Number of import-family contracts; the dense per-stage import-column stride.
    pub(crate) n_contract_import: usize,
    /// Number of export-family contracts; the dense per-stage export-column stride.
    pub(crate) n_contract_export: usize,
    /// Target hydro ID → system indices of hydros diverting to it (each hydro `d`
    /// with `diversion.downstream_id == target_id`).
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
    /// Per-plant `lead_stages` (`K_i`), length `n_anticipated`, anticipated-local order.
    pub(crate) anticipated_lead_stages: Vec<usize>,
    /// Anticipated-local position → global thermal index, length `n_anticipated`.
    pub(crate) anticipated_thermal_indices: Vec<usize>,
    /// Per-plant commissioning window `(entry_stage_id, exit_stage_id)`, length
    /// `n_anticipated`, anticipated-local order. The decision gate keys on the
    /// DELIVERY stage's operation window
    /// ([`StateLayout::is_anticipated_decision_active`]).
    pub(crate) anticipated_windows: Vec<(Option<i32>, Option<i32>)>,
    /// `study_stage_ids[t] = stage.id`, length `n_study_stages`. The decision gate
    /// keys its window clause on the delivery stage's `stage.id`, NOT the stage
    /// index, mapping delivery index `t + K_i` through this slice.
    pub(crate) study_stage_ids: Vec<i32>,
    /// Whether any penalty method is active.
    pub(crate) has_penalty: bool,
    /// Present-value multiplier at each stage, length `n_study_stages`. The strict
    /// predicate `stage_idx + K_i < n_stages` keeps every delivery lookup in range.
    pub(crate) cumulative_discount_factors: Vec<f64>,
    /// Σ `block.duration_hours` per study stage, length `n_study_stages` (same
    /// in-range guarantee as `cumulative_discount_factors`).
    pub(crate) total_hours_per_stage: Vec<f64>,
    /// Per-stage minimum target-storage trajectory, keyed `(hydro_idx, stage_id)
    /// → V_target` \[hm³\]. Computed once by a backward fold from the dead volume
    /// because the fold needs the full per-stage ζ·rate schedule across a hydro's
    /// Filling stages; the forbidden alternative — recomputing inside the per-stage
    /// `fill_filling_target_rows` (which sees one stage) — is wrong or re-walks the
    /// schedule on the hot path. `BTreeMap` for determinism, see `hydro_pos`.
    /// Empty for a non-filling build (parity-neutral). See
    /// [`build_filling_v_target`](super::template::build_filling_v_target).
    pub(crate) filling_v_target: BTreeMap<(usize, i32), f64>,
    /// Per-declared-arc resolved stage-clock weights, keyed by the arc's
    /// upstream hydro system index (parallel-mode fill; [`Self::arc_spread_chrono`]
    /// carries the chronological block-resolved factors). Absent for an
    /// undeclared arc — the fill's `k_0 = 1`, no-deposit branch. See
    /// [`build_arc_spread_k`](crate::setup::bucket_topology::build_arc_spread_k).
    pub(crate) arc_spread_k: HashMap<usize, Vec<Vec<f64>>>,
    /// Per-declared-arc, per-chronological-stage full [`SpreadResolution`]
    /// (`block_deposits`/`within_stage_routing`/`arrival_density`), keyed
    /// like [`Self::arc_spread_k`].
    /// `by_stage[stage_idx]` is `None` when that study stage's own `block_mode`
    /// is `Parallel` (the parallel fill reads [`Self::arc_spread_k`] instead).
    /// See [`build_arc_spread_chrono`](crate::setup::bucket_topology::build_arc_spread_chrono).
    pub(crate) arc_spread_chrono: HashMap<usize, Vec<Option<SpreadResolution>>>,
    /// `per_stage_mask[stage_idx]` holds the max reachable lag per declared
    /// downstream plant, discovery order (mirrors
    /// [`TransitBucketTopology::per_plant_depth`](crate::setup::bucket_topology::TransitBucketTopology::per_plant_depth)).
    /// Gates which bucket-definition rows [`StageLayout::new`] emits; see
    /// [`crate::setup::bucket_topology::TransitBucketTopology::per_stage_mask`].
    pub(crate) per_stage_mask: Vec<Vec<usize>>,
}

/// Column/row offsets for one stage's anticipated-thermal layout.
pub(crate) struct AnticipatedLayout {
    /// Start of anticipated-decision columns (one per plant, stage-level):
    /// `col_anticipated_decision_start + local_anticipated_idx`. Equals
    /// `col_thermal_end`.
    pub(crate) col_anticipated_decision_start: usize,
    /// Start of the `anticipated_state_out` column block (one per plant,
    /// stage-level). Sourced from `StateLayout::anticipated_state_out.start`, so
    /// the offset is stage-invariant — keeping the global stage-0 cut map on the
    /// correct column at every stage regardless of this stage's block count.
    pub(crate) col_anticipated_state_out_start: usize,
    /// Start of the `anticipated_state_out_def` equality row block. One row per
    /// ACTIVE plant (strict gate `stage_idx + K_p < n_stages`); inactive plants
    /// emit no row. Immediately after `row_anticipated_fishing_start`.
    pub(crate) row_anticipated_state_out_def_start: usize,
    /// Count of plants with `stage_idx + K_p < n_stages` (strict gate); drives the
    /// active-row iteration.
    // Rationale: read only by cross-module `debug_assert_eq!` guards in the matrix-fill
    // helpers; dead_code fires because the lint does not see cross-module field access.
    #[allow(dead_code)]
    pub(crate) n_anticipated_state_out_def_rows: usize,
    /// Start of anticipated-fishing rows (one per plant, always-active):
    /// `row_anticipated_fishing_start + local_idx`. After operational-violation rows.
    pub(crate) row_anticipated_fishing_start: usize,
    /// Anticipated-fishing row count; equals `n_anticipated` (always-active).
    pub(crate) n_anticipated_fishing_rows: usize,
}

/// Pre-computed column and row layout offsets for a single stage LP.
///
/// Owns the role-(b) geometry (per-stage equipment / slack / row ranges and the
/// entity counts that stride them) as its own fields, computed in
/// [`StageLayout::new`] anchored at the handle's
/// [`StateLayout::control_region_start`]. The stage-invariant role-(a) state
/// region is NOT duplicated here — it is read through the borrowed [`Self::state`]
/// handle. The control region begins at `state.control_region_start()`
/// (`theta + 1`), so the two regions meet there with no overlap.
pub(crate) struct StageLayout<'a> {
    /// Borrowed handle to the stage-invariant role-(a) state layout; the role-(a)
    /// accessors read through it rather than re-deriving offsets per stage. The
    /// dependency is one-directional (geometry → `StateLayout`), never the reverse.
    pub(crate) state: &'a StateLayout,
    /// Block count for this stage.
    pub(crate) n_blks: usize,
    /// Hydro count.
    pub(crate) n_h: usize,
    /// PAR lag order.
    pub(crate) lag_order: usize,
    /// Number of anticipated thermals (mirrors `TemplateBuildCtx.n_anticipated`).
    pub(crate) n_anticipated: usize,
    /// Maximum `lead_stages` across the anticipated thermals (`K_max`).
    pub(crate) k_max: usize,
    /// Anticipated-state ring-buffer width: `n_anticipated * k_max`.
    // Rationale: asserted only in layout unit tests; production helpers derive
    // `n_anticipated * k_max` inline, so dead_code fires on the production side.
    #[allow(dead_code)]
    pub(crate) n_ant_state: usize,
    /// Anticipated-thermal column/row offsets (see [`AnticipatedLayout`]).
    pub(crate) anticipated: AnticipatedLayout,
    /// Start of NCS generation columns (one per NCS per block, dense and
    /// system-indexed): `col_ncs_start + ncs_sys_idx * n_blks + blk`. A
    /// commissioning-dormant NCS keeps its column zeroed to `[0, 0]`, so the
    /// position is the entity's system index, not an active-local index.
    pub(crate) col_ncs_start: usize,
    /// Full NCS count (identical at every stage).
    pub(crate) n_ncs: usize,
    /// Start of pumping-flow columns (one per station per block, dense and
    /// system-indexed, block-major): `col_pumping_start + p_sys * n_blks + blk`. A
    /// dormant station keeps its column zeroed to `[0, 0]`; with `n_pumping == 0`
    /// the block is empty and `col_pumping_start == col_ncs_end`.
    pub(crate) col_pumping_start: usize,
    /// Full station count (identical at every stage); contributes `n_blks` columns
    /// each. Read into the scalar `StageTemplates::n_pumping` that bounds the
    /// per-(station, block) simulation primal read.
    pub(crate) n_pumping: usize,
    /// Start of import-contract columns (one per import contract per block,
    /// block-major): `col_contract_import_start + import_idx * n_blks + blk`. The
    /// import block follows pumping; with `n_contract_import == 0` it is empty and
    /// `col_contract_import_start == col_pumping_end`.
    pub(crate) col_contract_import_start: usize,
    /// Full import-contract count (identical at every stage).
    pub(crate) n_contract_import: usize,
    /// Start of export-contract columns (one per export contract per block,
    /// block-major): `col_contract_export_start + export_idx * n_blks + blk`. The
    /// export block follows the import block; with `n_contract_export == 0` it is
    /// empty and `col_contract_export_start == col_contract_import_end`.
    pub(crate) col_contract_export_start: usize,
    /// Full export-contract count (identical at every stage).
    pub(crate) n_contract_export: usize,
    /// Total column count.
    pub(crate) num_cols: usize,
    /// Start of generic constraint rows (one per active `(constraint, block)` pair),
    /// after operational-violation rows.
    pub(crate) row_generic_start: usize,
    /// Total row count.
    pub(crate) num_rows: usize,
    /// Generic constraint row count.
    pub(crate) n_generic_rows: usize,
    /// Structural dual-relevant row prefix; `0` (state pinning uses column bounds).
    pub(crate) n_dual_relevant: usize,
    /// `total_stage_hours * M3S_TO_HM3`; the water-balance noise/inflow scale.
    pub(crate) zeta: f64,
    /// Indices (into `ctx.hydros`) of hydros using FPHA at this stage.
    pub(crate) fpha_hydro_indices: Vec<usize>,
    /// Inverse of `fpha_hydro_indices`: system hydro index → FPHA-local index,
    /// length `n_h` (`None` at non-FPHA hydros). Single owner of the reverse map,
    /// read by the matrix-fill helpers in place of rebuilding it per call.
    pub(crate) fpha_local_index: Vec<Option<usize>>,
    /// Hyperplane count per FPHA hydro at this stage.
    pub(crate) fpha_planes_per_hydro: Vec<usize>,
    /// Indices (into `ctx.hydros`) of hydros with linearized evaporation at this stage.
    pub(crate) evap_hydro_indices: Vec<usize>,
    /// Per-row metadata for active generic constraint rows, one per active
    /// `(constraint, block)` pair in constraint-index-major order.
    pub(crate) generic_constraint_rows: Vec<GenericConstraintRowEntry>,

    // ── Role-(b) equipment column ranges (own fields) ────────────────────────
    // Empty families normalise to `0..0`; the empty-block-cursor accessors
    // therefore read a dedicated cursor field, not the `0` a bare `range.start`
    // would return for a collapsed range.
    /// Column range for the interior storage boundaries `S¹ … Sᴷ⁻¹` (one column
    /// per `(hydro, interior boundary)`, block-minor); empty `0..0` in parallel
    /// mode and when `K = 1`.
    // The Range read site lands with the per-block water-balance fill (which iterates
    // it to address interior columns); the bounds loop and accessor address interiors
    // through `storage_internal_start`, not this field. Until then only the layout
    // unit test reads the Range.
    #[allow(dead_code)]
    pub(crate) storage_internal: Range<usize>,
    /// Control-region anchor for `storage_internal` (= `control_region_start()`),
    /// read even when the family is empty. Within-family address is
    /// `storage_internal_start + h * (n_blks − 1) + (k − 1)` for interior boundary
    /// `k ∈ 1..n_blks` — stride `n_blks − 1`, not `n_blks`.
    pub(crate) storage_internal_start: usize,
    /// Column range for turbined flow (one per hydro per block).
    pub(crate) turbine: Range<usize>,
    /// Column range for spillage (one per hydro per block).
    pub(crate) spillage: Range<usize>,
    /// Column range for diversion flow (one per hydro per block).
    pub(crate) diversion: Range<usize>,
    /// Column range for thermal generation (one per thermal per block).
    pub(crate) thermal: Range<usize>,
    /// Column range for forward line flow (one per line per block).
    pub(crate) line_fwd: Range<usize>,
    /// Column range for reverse line flow (one per line per block).
    pub(crate) line_rev: Range<usize>,
    /// Column range for bus deficit variables (`B * S * K` columns).
    pub(crate) deficit: Range<usize>,
    /// Maximum deficit segments across buses (`S`); the deficit-stride constant.
    pub(crate) max_deficit_segments: usize,
    /// Column range for bus excess variables (one per bus per block).
    pub(crate) excess: Range<usize>,
    /// Column range for import-contract variables (one per import contract per
    /// block); empty `start..start` at `col_pumping_end` with no import contracts.
    pub(crate) contract_import: Range<usize>,
    /// Column range for export-contract variables (one per export contract per
    /// block); empty `start..start` at the import-block end with no export contracts.
    pub(crate) contract_export: Range<usize>,
    /// Column range for inflow non-negativity slack (one per hydro, stage-level);
    /// `0..0` without the penalty or hydros. Stored first-class so the per-stage
    /// simulation geometry reads the stage-correct range — a single global stage-0
    /// range would shift under a non-uniform block schedule.
    pub(crate) inflow_slack: Range<usize>,
    /// Column-block cursor at which the FPHA generation block begins, even when
    /// that block is empty (`inflow_slack.end` with penalty, else `excess.end`).
    pub(crate) generation_col_start: usize,
    /// Column range for FPHA generation (one per FPHA hydro per block).
    pub(crate) generation: Range<usize>,
    /// Column-block cursor at which the evaporation block begins, even when empty
    /// (`generation_col_start + n_fpha_hydros * n_blks`).
    pub(crate) evap_col_start: usize,
    /// Per-`(evaporation hydro, block)` column/row indices, block-major
    /// (`local * n_blks + blk`). At `n_blks == 1` the slot for evap hydro `i` is
    /// `i`, parallel to `evap_hydro_indices`.
    pub(crate) evap_indices: Vec<EvaporationIndices>,
    /// Column range for under-withdrawal slack (one per hydro). `0..0` with no
    /// hydros.
    pub(crate) withdrawal_slack_neg: Range<usize>,
    /// Column range for over-withdrawal slack (one per hydro).
    pub(crate) withdrawal_slack_pos: Range<usize>,
    /// Column range for outflow-below-minimum slack (one per hydro per block).
    pub(crate) outflow_below_slack: Range<usize>,
    /// Column range for outflow-above-maximum slack (one per hydro per block).
    pub(crate) outflow_above_slack: Range<usize>,
    /// Column range for turbine-below-minimum slack (one per hydro per block).
    pub(crate) turbine_below_slack: Range<usize>,
    /// Column range for generation-below-minimum slack (one per hydro per block).
    pub(crate) generation_below_slack: Range<usize>,
    /// Shared post-equipment column cursor for empty-hydro fallbacks
    /// (`evap_col_start`). The eight withdrawal/operational column families and
    /// the NCS region collapse onto this single cursor when `n_h == 0`.
    pub(crate) post_equipment_col_start: usize,

    // ── Role-(b) constraint row ranges (own fields) ──────────────────────────
    /// Row index of the first z-inflow definition constraint. Row 0; state pinning
    /// uses column bounds, so no state-fixing rows precede the z-inflow block.
    pub(crate) z_inflow_row_start: usize,
    /// Row range for water balance constraints: `n_h` rows in parallel mode,
    /// `n_h * n_blks` in chronological mode (the `K` chained per-hydro rows). Rows
    /// are block-major like `load_balance`: the row for `(h, k)` is
    /// `water_balance.start + h * n_blks + k` (entity-outer, block-inner) via
    /// [`BlockGrid::flat`](crate::indexer::BlockGrid::flat); the transposed
    /// `k * n_h + h` is the wrong-but-compiling alternative.
    pub(crate) water_balance: Range<usize>,
    /// Row range for travel-time bucket definition rows: `b_d^out − b_{d+1}^in
    /// − deposit_d = 0`, one row per (plant, lag) bucket REACHABLE at this
    /// stage (`state.transit_bucket_column_order[slot]`'s lag within this stage's
    /// `per_stage_mask` cap for that plant — see [`Self::transit_bucket_row_pos`]);
    /// unlike `anticipated_state`'s active-plant sparseness, a lag beyond the
    /// cap targets a stage outside `[0, n_stages)` and gets no row at ANY
    /// stage from here to the horizon (the cap only shrinks). Placed
    /// immediately after [`Self::water_balance`], so `load_balance` and every
    /// row cursor after it shift by this stage's reachable count (`<=
    /// state.n_buckets`, `== state.n_buckets` only while every lag is still
    /// within-horizon). Empty `start..start` when `state.n_buckets == 0` (the
    /// B==0 byte-identity anchor: `load_balance` collapses back onto
    /// `water_balance.end`).
    pub(crate) transit_bucket_definition: Range<usize>,
    /// For each GLOBAL bucket index (`state.transit_bucket_column_order`'s index),
    /// this stage's compact row position within [`Self::transit_bucket_definition`], or
    /// `None` when its lag is beyond this stage's reachable cap (no row; the
    /// matching deposit in [`super::entries`]'s arc-release fill is dropped,
    /// not misdirected to another row). Length `state.n_buckets`.
    pub(crate) transit_bucket_row_pos: Vec<Option<usize>>,
    /// Row range for load balance constraints (one per bus per block).
    pub(crate) load_balance: Range<usize>,
    /// Row cursor at which the evaporation row block begins (`fpha_rows_end`),
    /// even when the FPHA block is empty.
    pub(crate) fpha_rows_end: usize,
    /// Row range for min-outflow constraints (one per hydro per block).
    pub(crate) min_outflow_rows: Range<usize>,
    /// Row range for max-outflow constraints (one per hydro per block).
    pub(crate) max_outflow_rows: Range<usize>,
    /// Row range for min-turbine constraints (one per hydro per block).
    pub(crate) min_turbine_rows: Range<usize>,
    /// Row range for min-generation constraints (one per hydro per block).
    pub(crate) min_generation_rows: Range<usize>,
    /// Shared post-equipment row cursor for empty-hydro fallbacks
    /// (`fpha_rows_end + n_evap_hydros`). The four operational-violation row
    /// families collapse onto this single cursor when `n_h == 0`.
    pub(crate) post_equipment_row_start: usize,
    /// First per-stage `σ_fill`-target row (one per Filling-phase filling hydro);
    /// after the operational-violation rows, in the pre-cut region. Empty at every
    /// non-Filling stage. MUST stay strictly below `num_rows`: a row at index
    /// `>= num_rows` aliases the append-only cut rows (slot-identity warm-start
    /// matches cut rows from `num_rows`) and corrupts every cut.
    pub(crate) row_filling_target_start: usize,
    /// First `σ_fill` slack column (one per Filling-phase filling hydro); the
    /// second-to-last per-stage column family, after generic-slack and before
    /// `filled_min_storage_floor`. Empty for a non-filling system, leaving prior
    /// `col_*_start` and `num_cols` byte-identical.
    pub(crate) col_filling_target_start: usize,
    /// System hydro indices emitting a `σ_fill` target at this stage, ascending.
    /// Parallel to both the `filling_target` row and `σ_fill` column blocks: local
    /// index `i` → row `row_filling_target_start + i`, column
    /// `col_filling_target_start + i`.
    pub(crate) filling_target_hydro_indices: Vec<usize>,
    /// First soft `σ^{v-}` operating-floor row (one per Operating-phase filling
    /// hydro); sibling to `filling_target` in the pre-cut region. Same
    /// `row >= num_rows` aliasing invariant as `row_filling_target_start`.
    pub(crate) row_filled_min_storage_floor_start: usize,
    /// First soft `σ^{v-}` slack column (one per Operating-phase filling hydro); the
    /// LAST per-stage column family, so its presence cannot shift any other family.
    /// Empty for a non-filling system, leaving other `col_*_start`/`num_cols`
    /// byte-identical.
    pub(crate) col_filled_min_storage_floor_start: usize,
    /// System hydro indices emitting a `σ^{v-}` floor at this stage, ascending.
    /// Parallel to both the `filled_min_storage_floor` row and column blocks. DISTINCT
    /// from `filling_target_hydro_indices` (`σ_fill`, Filling phase); the two
    /// never overlap (Operating vs Filling).
    pub(crate) filled_min_storage_floor_hydro_indices: Vec<usize>,

    // ── Role-(b) anticipated identity maps (own fields) ──────────────────────
    /// Reverse map: global thermal position → anticipated-local index. Built once
    /// for O(1) resolution in the generic-constraint `AnticipatedDecision` arm.
    pub(crate) anticipated_local_by_sys_pos: HashMap<usize, usize>,
}

// ── Private helper return structs ─────────────────────────────────────────────

/// Layout metadata for all active generic constraint rows and slack columns.
struct GenericConstraintLayout {
    n_generic_rows: usize,
    n_generic_slack_cols: usize,
    generic_constraint_rows: Vec<GenericConstraintRowEntry>,
}

/// Column and row ranges for the four operational-violation slack families,
/// non-empty only when `n_h > 0`. Slack columns follow the withdrawal slacks;
/// constraint rows follow the evaporation rows.
struct OperViolationRanges {
    outflow_below_slack: Range<usize>,
    outflow_above_slack: Range<usize>,
    turbine_below_slack: Range<usize>,
    generation_below_slack: Range<usize>,
    min_outflow_rows: Range<usize>,
    max_outflow_rows: Range<usize>,
    min_turbine_rows: Range<usize>,
    min_generation_rows: Range<usize>,
}

impl OperViolationRanges {
    /// The all-empty (`0..0`) variant used when `n_h == 0`.
    fn empty() -> Self {
        Self {
            outflow_below_slack: 0..0,
            outflow_above_slack: 0..0,
            turbine_below_slack: 0..0,
            generation_below_slack: 0..0,
            min_outflow_rows: 0..0,
            max_outflow_rows: 0..0,
            min_turbine_rows: 0..0,
            min_generation_rows: 0..0,
        }
    }
}

/// Row cursor where the evaporation rows begin (after the FPHA rows):
/// `start_row + n_blks * Σ planes`. The per-hydro
/// [`FphaRowRange`](crate::indexer::FphaRowRange) entries live on
/// `StageData.indexer`, not here.
fn build_fpha_rows(planes_per_hydro: &[usize], n_blks: usize, start_row: usize) -> usize {
    let total_planes: usize = planes_per_hydro.iter().sum();
    start_row + n_blks * total_planes
}

/// For each entry of `column_order` (global bucket index `slot`, `(plant, lag)`),
/// this stage's compact position within [`StageLayout::transit_bucket_definition`], or
/// `None` when `lag` exceeds `per_stage_mask[stage_idx]`'s max reachable lag
/// for that plant. `column_order` groups contiguously by plant in the SAME
/// discovery order `per_stage_mask` indexes
/// ([`crate::setup::bucket_topology::build_transit_bucket_topology`]), so a plant
/// transition in the scan advances the mask index. Returns the mapping and the
/// reachable count (`transit_bucket_definition`'s row length).
fn build_transit_bucket_row_pos(
    column_order: &[(usize, usize)],
    per_stage_mask: &[Vec<usize>],
    stage_idx: usize,
) -> (Vec<Option<usize>>, usize) {
    if column_order.is_empty() {
        // B==0 byte-identity anchor: no declared bucket, so no per-stage mask
        // entry is required (`per_stage_mask` may be empty in fixtures that
        // never build one).
        return (Vec::new(), 0);
    }
    let stage_mask = &per_stage_mask[stage_idx];
    let mut transit_bucket_row_pos = Vec::with_capacity(column_order.len());
    let mut plant_group = 0_usize;
    let mut prev_plant: Option<usize> = None;
    let mut n_reachable = 0_usize;
    for &(plant_idx, lag) in column_order {
        if prev_plant != Some(plant_idx) {
            if prev_plant.is_some() {
                plant_group += 1;
            }
            prev_plant = Some(plant_idx);
        }
        if lag <= stage_mask[plant_group] {
            transit_bucket_row_pos.push(Some(n_reachable));
            n_reachable += 1;
        } else {
            transit_bucket_row_pos.push(None);
        }
    }
    (transit_bucket_row_pos, n_reachable)
}

/// Evaporation column/row indices per `(evaporation hydro, block)`, block-major
/// (`local * n_blks + blk`) to mirror the block-strided generation columns.
/// Within-triple columns at [`EVAP_FLOW_OFFSET`] / [`EVAP_F_PLUS_OFFSET`] /
/// [`EVAP_F_MINUS_OFFSET`], strided by [`EVAP_COLS_PER_HYDRO`]; one row per
/// `(hydro, block)`. At `n_blks == 1` the slot for hydro `i` is `i`, so a reader
/// indexing by the hydro-local index alone still lands on block 0.
fn build_evap_indices(
    n_evap_hydros: usize,
    n_blks: usize,
    col_start: usize,
    row_start: usize,
) -> Vec<EvaporationIndices> {
    let mut out = Vec::with_capacity(n_evap_hydros * n_blks);
    for i in 0..n_evap_hydros {
        for blk in 0..n_blks {
            let slot = i * n_blks + blk;
            let triple_base = col_start + slot * EVAP_COLS_PER_HYDRO;
            out.push(EvaporationIndices {
                evaporation_flow_col: triple_base + EVAP_FLOW_OFFSET,
                f_evap_plus_col: triple_base + EVAP_F_PLUS_OFFSET,
                f_evap_minus_col: triple_base + EVAP_F_MINUS_OFFSET,
                evap_row: row_start + slot,
            });
        }
    }
    out
}

// ── Private helper functions ───────────────────────────────────────────────────

/// Collect the FPHA hydro indices and per-hydro plane counts for this stage.
///
/// A filling hydro is dropped from the FPHA set in `PreFilling` **or** `Filling`:
/// a non-operating plant has zero productivity, and the operating-range hyperplane
/// fit is invalid below `min_storage` where a filling reservoir sits. Because the
/// generation column block is densely packed by FPHA-local index, dropping a hydro
/// here removes its column entirely — no orphaned `[0, max]` column for an
/// unconstrained solve to exploit. `stage_id` is the study `stage.id`, not the
/// stage index ([`filling_phase`] keys on the commissioning id). A
/// commissioning-dormant non-filling hydro is `PreFilling` and is dropped here too;
/// a non-filling hydro with no window is `Operating` at every stage (parity-neutral).
fn identify_fpha_hydros(
    ctx: &TemplateBuildCtx<'_>,
    stage_idx: usize,
    stage_id: i32,
) -> (Vec<usize>, Vec<usize>) {
    let mut fpha_hydro_indices: Vec<usize> = Vec::new();
    let mut fpha_planes_per_hydro: Vec<usize> = Vec::new();
    for h_idx in 0..ctx.n_hydros {
        let hydro = &ctx.hydros[h_idx];
        if matches!(
            filling_phase(
                hydro.filling.as_ref(),
                hydro.entry_stage_id,
                hydro.exit_stage_id,
                stage_id
            ),
            Phase::PreFilling | Phase::Filling
        ) {
            continue;
        }
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
/// A hydro is dropped from the evaporation set only in `PreFilling` (before
/// `start_stage_id`, or while a non-filling hydro is commissioning-dormant, the dam
/// and hence the reservoir surface does not exist). Evaporation is **kept** during
/// `Filling` — the opposite of the FPHA rule (excluded in `PreFilling` *and*
/// `Filling`); the two must not be unified. A non-filling hydro with no window is
/// `Operating` at every stage (parity-neutral).
fn identify_evap_hydros(ctx: &TemplateBuildCtx<'_>, stage_id: i32) -> Vec<usize> {
    (0..ctx.n_hydros)
        .filter(|&h_idx| {
            let hydro = &ctx.hydros[h_idx];
            if matches!(
                filling_phase(
                    hydro.filling.as_ref(),
                    hydro.entry_stage_id,
                    hydro.exit_stage_id,
                    stage_id
                ),
                Phase::PreFilling
            ) {
                return false;
            }
            matches!(
                ctx.evaporation_models.model(h_idx),
                EvaporationModel::Linearized { .. }
            )
        })
        .collect()
}

/// Collect the indices of hydros emitting a per-stage `σ_fill` target at this
/// stage: the filling hydros (`filling.is_some()`) in [`Phase::Filling`].
///
/// EVERY Filling stage carries a floor, NOT only the terminal stage at `entry −
/// 1`: the per-stage trajectory `V_target[t]` requires one soft floor `v_out[t] +
/// σ_fill[t] ≥ V_target[t]` at each. The wrong-but-compiling alternative —
/// restricting membership to `entry − 1 == stage_id` (the v1 terminal-only rule) —
/// drops every intermediate floor. `PreFilling`/`Operating` are excluded by
/// [`filling_phase`] (`filled_min_storage_floor` takes over at/after `entry`). A
/// non-filling hydro is `Operating` at every stage (parity-neutral).
fn identify_filling_target_hydros(ctx: &TemplateBuildCtx<'_>, stage_id: i32) -> Vec<usize> {
    (0..ctx.n_hydros)
        .filter(|&h_idx| {
            let hydro = &ctx.hydros[h_idx];
            hydro.filling.is_some()
                && matches!(
                    filling_phase(
                        hydro.filling.as_ref(),
                        hydro.entry_stage_id,
                        hydro.exit_stage_id,
                        stage_id
                    ),
                    Phase::Filling
                )
        })
        .collect()
}

/// Collect the indices of hydros emitting a soft `σ^{v-}` operating-floor at this
/// stage: the filling hydros (`filling.is_some()`) in [`Phase::Operating`].
///
/// DISTINCT from [`identify_filling_target_hydros`] (`σ_fill`): `σ^{v-}` fires at
/// EVERY Operating stage, `σ_fill` at EVERY Filling stage; the two never overlap
/// and carry different costs.
///
/// The soft floor is scoped to filling hydros DELIBERATELY — a non-filling
/// `Operating` hydro keeps its hard `min_storage` floor (same gate as the relax in
/// [`super::columns::fill_storage_columns`]). The wrong-but-compiling alternative —
/// a GLOBAL soft floor matching every Operating hydro regardless of `filling` —
/// would let the optimizer cheaply violate dead volume system-wide. Empty for a
/// non-filling build (parity-neutral).
fn identify_filled_min_storage_floor_hydros(
    ctx: &TemplateBuildCtx<'_>,
    stage_id: i32,
) -> Vec<usize> {
    (0..ctx.n_hydros)
        .filter(|&h_idx| {
            let hydro = &ctx.hydros[h_idx];
            matches!(
                filling_phase(
                    hydro.filling.as_ref(),
                    hydro.entry_stage_id,
                    hydro.exit_stage_id,
                    stage_id
                ),
                Phase::Operating
            ) && hydro.filling.is_some()
        })
        .collect()
}

/// Allocate the slack column index/indices for one generic-constraint row,
/// advancing `n_slack_cols`: zero columns when slack is disabled, one for
/// inequality, two (plus then minus) for equality.
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
/// One [`GenericConstraintRowEntry`] per active `(constraint, block)` pair, except
/// a `block_id = None` bound over a block-independent expression, which collapses
/// to a single stage-level row.
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

        // A block-independent expression produces identical rows for every block,
        // so it collapses to one stage-level row priced by total hours.
        let collapse_stage_level =
            crate::generic_constraints::expression_is_block_independent(&constraint.expression);

        // Bind the constraint-invariant fields once; the closure keeps the three
        // arms below field-for-field identical (only per-row fields vary).
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
                    // block_id is a non-negative 0-indexed block position (upstream
                    // validation), so the cast_sign_loss is safe.
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

impl<'a> StageLayout<'a> {
    // Rationale: too_many_lines — the role-(b) ranges derive sequentially from the
    // previous range's `.end`; keeping the whole chain in one function is what makes the
    // sequential-offset contract auditable in a single linear read.
    // Rationale: similar_names — `state` (the handle, matching `StageData.state`) next to
    // `stage`/`stage_idx`; both names are established, renaming would obscure intent.
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    pub(crate) fn new(
        ctx: &TemplateBuildCtx<'_>,
        state: &'a StateLayout,
        stage: &Stage,
        stage_idx: usize,
    ) -> Self {
        let n_blks = stage.blocks.len();
        let n_h = ctx.n_hydros;

        // Per-stage membership sets that size the column/row blocks below; each gates
        // filling hydros by `stage.id` (see the `identify_*` helpers).
        let (fpha_hydro_indices, fpha_planes_per_hydro) =
            identify_fpha_hydros(ctx, stage_idx, stage.id);
        let evap_hydro_indices = identify_evap_hydros(ctx, stage.id);
        let filling_target_hydro_indices = identify_filling_target_hydros(ctx, stage.id);
        let filled_min_storage_floor_hydro_indices =
            identify_filled_min_storage_floor_hydros(ctx, stage.id);

        let mut fpha_local_index: Vec<Option<usize>> = vec![None; n_h];
        for (local_idx, &h_idx) in fpha_hydro_indices.iter().enumerate() {
            fpha_local_index[h_idx] = Some(local_idx);
        }

        // Strides the deficit column block.
        let max_deficit_segments = ctx
            .buses
            .iter()
            .map(|b| b.deficit_segments.len())
            .max()
            .unwrap_or(0);

        // ── Role-(b) equipment column ranges ─────────────────────────────────
        // Anchored at the handle's `control_region_start()` (the role-(a)/role-(b)
        // seam); each range starts at the previous range's `.end`, strided by THIS
        // stage's `n_blks` (the per-stage authority over the stage-0 global stride).
        let n_interior = match stage.block_mode {
            BlockMode::Chronological => n_blks.saturating_sub(1),
            BlockMode::Parallel => 0,
        };
        let storage_internal_start = state.control_region_start();
        let storage_internal_end = storage_internal_start + n_h * n_interior;
        let turbine_start = storage_internal_end;
        let spillage_start = turbine_start + n_h * n_blks;
        let diversion_start = spillage_start + n_h * n_blks;
        let thermal_start = diversion_start + n_h * n_blks;
        let thermal_end = thermal_start + ctx.n_thermals * n_blks;
        // Anticipated-decision columns occupy `[thermal_end, thermal_end + A)`;
        // `line_fwd` follows them.
        let anticipated_decision_end = thermal_end + ctx.n_anticipated;
        let line_fwd_start = anticipated_decision_end;
        let line_rev_start = line_fwd_start + ctx.n_lines * n_blks;
        let deficit_start = line_rev_start + ctx.n_lines * n_blks;
        let excess_start = deficit_start + ctx.n_buses * max_deficit_segments * n_blks;
        let excess_end = excess_start + ctx.n_buses * n_blks;

        // Inflow non-negativity slack, after `excess`, only with the penalty.
        let has_inflow_penalty = ctx.has_penalty && n_h > 0;
        let inflow_slack = if has_inflow_penalty {
            excess_end..excess_end + n_h
        } else {
            0..0
        };

        // FPHA generation columns follow `inflow_slack` (or `excess`).
        // `generation_col_start` is the empty-block cursor `col_generation_start`
        // reads when the block is empty, so it never returns a bare-`.start` `0`.
        let n_fpha_hydros = fpha_hydro_indices.len();
        let generation_col_start = if has_inflow_penalty {
            inflow_slack.end
        } else {
            excess_end
        };
        let generation_end = generation_col_start + n_fpha_hydros * n_blks;
        let generation = if n_fpha_hydros > 0 {
            generation_col_start..generation_end
        } else {
            0..0
        };

        // Evaporation columns after FPHA generation; `evap_col_start` is the
        // empty-block cursor `col_evap_start` reads. One `EVAP_COLS_PER_HYDRO`
        // triple per `(evap hydro, block)`, so the block grows by `n_blks` in
        // chronological mode (`n_blks == 1` leaves it at the single-triple size).
        let n_evap_hydros = evap_hydro_indices.len();
        let evap_col_start = generation_end;
        let evap_col_end = evap_col_start + n_evap_hydros * n_blks * EVAP_COLS_PER_HYDRO;
        let post_equipment_col_start = evap_col_start;

        // ── Role-(b) constraint row ranges ───────────────────────────────────
        // z_inflow rows start at row 0 — state pinning uses column bounds, so no
        // state-fixing row range precedes them.
        let z_inflow_row_start = 0_usize;
        let water_balance_start = z_inflow_row_start + n_h;
        let n_water_blocks = match stage.block_mode {
            BlockMode::Chronological => n_blks,
            BlockMode::Parallel => 1,
        };
        let n_water_rows = n_h * n_water_blocks;
        let water_balance = water_balance_start..water_balance_start + n_water_rows;
        // Sized from this stage's reachable count, not the stage-invariant
        // `state.n_buckets`: `build_transit_bucket_row_pos` masks a lag beyond
        // `ctx.per_stage_mask[stage_idx]`'s per-plant cap out of the row range
        // entirely (`horizon_cap_active`'s "dropped by construction").
        let transit_bucket_definition_start = water_balance.end;
        let (transit_bucket_row_pos, n_transit_bucket_rows) = build_transit_bucket_row_pos(
            &state.transit_bucket_column_order,
            &ctx.per_stage_mask,
            stage_idx,
        );
        let transit_bucket_definition = transit_bucket_definition_start
            ..transit_bucket_definition_start + n_transit_bucket_rows;
        let load_balance_start = transit_bucket_definition.end;
        let load_balance_end = load_balance_start + ctx.n_buses * n_blks;
        let load_balance = load_balance_start..load_balance_end;

        // FPHA rows follow load_balance; only the end cursor is kept here (the
        // per-hydro ranges live on `StageData.indexer`). `fpha_rows_end` is the
        // evaporation-row start even when the FPHA block is empty.
        let fpha_rows_end = build_fpha_rows(&fpha_planes_per_hydro, n_blks, load_balance_end);

        // Evaporation rows follow FPHA rows; `evap_rows_end` is the post-equipment
        // row cursor the empty operational-violation row families collapse onto.
        // One row per `(evap hydro, block)`, so the block grows by `n_blks` — the
        // cursor chain below MUST stay in lockstep or every downstream row shifts.
        let evap_indices = build_evap_indices(n_evap_hydros, n_blks, evap_col_start, fpha_rows_end);
        let evap_rows_end = fpha_rows_end + n_evap_hydros * n_blks;
        let post_equipment_row_start = evap_rows_end;

        // Withdrawal slacks + the four operational-violation slack families (after
        // the evaporation columns) and their matching rows (after the evaporation
        // rows). All collapse to `0..0` when `n_h == 0`, with downstream cursors
        // falling back to the post-equipment cursor so no stale offset survives.
        let (withdrawal_slack_neg, withdrawal_slack_pos, oper) = if n_h > 0 {
            let neg = evap_col_end..evap_col_end + n_h;
            let pos = neg.end..neg.end + n_h;
            let n_op = n_h * n_blks;
            let ob = pos.end..pos.end + n_op;
            let oa = ob.end..ob.end + n_op;
            let tb = oa.end..oa.end + n_op;
            let gb = tb.end..tb.end + n_op;
            let r_min_out = evap_rows_end..evap_rows_end + n_op;
            let r_max_out = r_min_out.end..r_min_out.end + n_op;
            let r_min_turb = r_max_out.end..r_max_out.end + n_op;
            let r_min_gen = r_min_turb.end..r_min_turb.end + n_op;
            (
                neg,
                pos,
                OperViolationRanges {
                    outflow_below_slack: ob,
                    outflow_above_slack: oa,
                    turbine_below_slack: tb,
                    generation_below_slack: gb,
                    min_outflow_rows: r_min_out,
                    max_outflow_rows: r_max_out,
                    min_turbine_rows: r_min_turb,
                    min_generation_rows: r_min_gen,
                },
            )
        } else {
            (0..0, 0..0, OperViolationRanges::empty())
        };

        let n_ant_state = ctx.n_anticipated * ctx.k_max;

        // NCS follows the last operational-violation slack family; when `n_h == 0`
        // that family is the empty `0..0` sentinel, so fall back to the
        // post-equipment cursor — a future family inserted before NCS then cannot
        // leave this `n_h == 0` start on a stale cursor.
        let n_ncs = ctx.non_controllable_sources.len();
        let col_ncs_start = if n_h > 0 {
            oper.generation_below_slack.end
        } else {
            post_equipment_col_start
        };
        let col_ncs_end = col_ncs_start + n_ncs * n_blks;

        // Row offsets: operational, σ_fill, σ^{v-}, fishing, state-out-def, generic.
        // n_dual_relevant is 0 — state pinning uses column bounds, so the cut path
        // reads view.reduced_costs, not a structural dual prefix.
        let n_dual_relevant = 0_usize;
        let n_op_rows = n_h * n_blks;
        // Anchor the fishing-row chain on the post-equipment cursor when `n_h == 0`
        // (the operational-violation blocks normalise to `0..0`), so the empty-hydro
        // row layout cannot drift onto a stale `> 0`-branch offset.
        let row_min_generation_start = if n_h > 0 {
            oper.min_generation_rows.start
        } else {
            post_equipment_row_start
        };

        // σ_fill then σ^{v-} rows, in the pre-cut region after the
        // operational-violation rows. Both MUST stay strictly below `num_rows`: a
        // row at index `>= num_rows` aliases the append-only cut rows (slot-identity
        // warm-start matches cut rows from `num_rows`) and corrupts every cut.
        let n_filling_target_rows = filling_target_hydro_indices.len();
        let row_filling_target_start = row_min_generation_start + n_op_rows;
        let n_filled_min_storage_floor_rows = filled_min_storage_floor_hydro_indices.len();
        let row_filled_min_storage_floor_start = row_filling_target_start + n_filling_target_rows;

        // One fishing row per anticipated plant (always-active).
        let n_anticipated_fishing_rows = ctx.n_anticipated;
        let row_anticipated_fishing_start =
            row_filled_min_storage_floor_start + n_filled_min_storage_floor_rows;

        // Anticipated-state-out definition rows: one per ACTIVE plant
        // (`StateLayout::is_anticipated_decision_active`, the single-owner gate).
        let n_stages = ctx.resolved.bounds.n_stages();
        let n_anticipated_state_out_def_rows = (0..ctx.n_anticipated)
            .filter(|&local_idx| {
                state.is_anticipated_decision_active(
                    local_idx,
                    stage_idx,
                    n_stages,
                    &ctx.anticipated_windows,
                    &ctx.study_stage_ids,
                )
            })
            .count();
        let row_anticipated_state_out_def_start =
            row_anticipated_fishing_start + n_anticipated_fishing_rows;
        let row_generic_start =
            row_anticipated_state_out_def_start + n_anticipated_state_out_def_rows;

        // Pumping columns follow NCS, before the generic-slack columns.
        let n_pumping = ctx.n_pumping;
        let col_pumping_start = col_ncs_end;
        let col_pumping_end = col_pumping_start + n_pumping * n_blks;

        // Contract columns follow pumping, before the generic-slack columns:
        // import block then export block. With both counts 0 the blocks are empty
        // and col_generic_slack_start stays at col_pumping_end (parity-neutral).
        let n_contract_import = ctx.n_contract_import;
        let n_contract_export = ctx.n_contract_export;
        let col_contract_import_start = col_pumping_end;
        let col_contract_import_end = col_contract_import_start + n_contract_import * n_blks;
        let col_contract_export_start = col_contract_import_end;
        let col_contract_export_end = col_contract_export_start + n_contract_export * n_blks;

        let col_generic_slack_start = col_contract_export_end;
        let generic =
            enumerate_generic_constraint_rows(ctx, stage, n_blks, col_generic_slack_start);

        // σ_fill then σ^{v-} are the last two per-stage column families; σ^{v-}
        // last so its presence cannot shift any other family's start.
        let col_filling_target_start = col_generic_slack_start + generic.n_generic_slack_cols;
        let col_filled_min_storage_floor_start =
            col_filling_target_start + filling_target_hydro_indices.len();
        let num_cols =
            col_filled_min_storage_floor_start + filled_min_storage_floor_hydro_indices.len();
        let num_rows = row_generic_start + generic.n_generic_rows;
        let zeta = stage.blocks.iter().map(|b| b.duration_hours).sum::<f64>() * M3S_TO_HM3;

        // The state-out cut-target column is sourced from its stage-invariant
        // state-region position (`state.anticipated_state_out.start`), NOT
        // `thermal_end + n_anticipated`, so the global stage-0 cut map lands on the
        // correct column even when this stage's block count differs from stage 0's.
        let col_anticipated_state_out_start = if ctx.n_anticipated > 0 {
            state.anticipated_state_out.start
        } else {
            thermal_end
        };
        let anticipated = AnticipatedLayout {
            col_anticipated_decision_start: thermal_end,
            col_anticipated_state_out_start,
            row_anticipated_state_out_def_start,
            n_anticipated_state_out_def_rows,
            row_anticipated_fishing_start,
            n_anticipated_fishing_rows,
        };

        // Reverse map for O(1) `AnticipatedDecision` generic-constraint resolution.
        let anticipated_local_by_sys_pos = ctx
            .anticipated_thermal_indices
            .iter()
            .enumerate()
            .map(|(local, &sys_pos)| (sys_pos, local))
            .collect();

        Self {
            state,
            n_blks,
            n_h,
            lag_order: ctx.max_par_order,
            n_anticipated: ctx.n_anticipated,
            k_max: ctx.k_max,
            n_ant_state,
            anticipated,
            col_ncs_start,
            n_ncs,
            col_pumping_start,
            n_pumping,
            col_contract_import_start,
            n_contract_import,
            col_contract_export_start,
            n_contract_export,
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
            storage_internal: storage_internal_start..storage_internal_end,
            storage_internal_start,
            turbine: turbine_start..spillage_start,
            spillage: spillage_start..diversion_start,
            diversion: diversion_start..thermal_start,
            thermal: thermal_start..thermal_end,
            line_fwd: line_fwd_start..line_rev_start,
            line_rev: line_rev_start..deficit_start,
            deficit: deficit_start..excess_start,
            max_deficit_segments,
            excess: excess_start..excess_end,
            contract_import: col_contract_import_start..col_contract_import_end,
            contract_export: col_contract_export_start..col_contract_export_end,
            inflow_slack,
            generation_col_start,
            generation,
            evap_col_start,
            evap_indices,
            withdrawal_slack_neg,
            withdrawal_slack_pos,
            outflow_below_slack: oper.outflow_below_slack,
            outflow_above_slack: oper.outflow_above_slack,
            turbine_below_slack: oper.turbine_below_slack,
            generation_below_slack: oper.generation_below_slack,
            post_equipment_col_start,
            z_inflow_row_start,
            water_balance,
            transit_bucket_definition,
            transit_bucket_row_pos,
            load_balance,
            fpha_rows_end,
            min_outflow_rows: oper.min_outflow_rows,
            max_outflow_rows: oper.max_outflow_rows,
            min_turbine_rows: oper.min_turbine_rows,
            min_generation_rows: oper.min_generation_rows,
            post_equipment_row_start,
            row_filling_target_start,
            col_filling_target_start,
            filling_target_hydro_indices,
            row_filled_min_storage_floor_start,
            col_filled_min_storage_floor_start,
            filled_min_storage_floor_hydro_indices,
            anticipated_local_by_sys_pos,
        }
    }

    /// Resolve a block-major LP column address: `start + entity * n_blks + blk`
    /// (entity is the OUTER stride factor, block the INNER offset). The transposed
    /// `blk * n_entities + entity` is the wrong-but-compiling alternative — same
    /// length, but it interleaves columns across entities and silently misbuilds the
    /// LP. Delegates to [`BlockGrid::flat`](crate::indexer::BlockGrid::flat), the
    /// single owner of the stride arithmetic.
    #[inline]
    pub(crate) fn block_col(&self, start: usize, entity: usize, blk: usize) -> usize {
        self.block_grid().flat(start, entity, blk)
    }

    /// The [`BlockGrid`] address primitive for this stage's LP, carrying this
    /// stage's own `n_blks` and `max_deficit_segments`.
    #[inline]
    #[must_use]
    pub(crate) fn block_grid(&self) -> BlockGrid {
        BlockGrid::new(self.n_blks, self.max_deficit_segments)
    }

    /// Whether anticipated plant `local_idx` is active at `stage_idx`. Delegates to
    /// the single-owner gate [`StateLayout::is_anticipated_decision_active`], so the
    /// active set is defined once across `new` and the column/row fills.
    #[inline]
    #[must_use]
    pub(crate) fn is_anticipated_decision_active(
        &self,
        local_idx: usize,
        stage_idx: usize,
        n_stages: usize,
        anticipated_windows: &[(Option<i32>, Option<i32>)],
        study_stage_ids: &[i32],
    ) -> bool {
        self.state.is_anticipated_decision_active(
            local_idx,
            stage_idx,
            n_stages,
            anticipated_windows,
            study_stage_ids,
        )
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

    /// Base column of the `(evap hydro local_idx, block blk)` triple, block-major
    /// (`(local_idx * n_blks + blk) * EVAP_COLS_PER_HYDRO`). Single owner of the
    /// evaporation block stride; the three offset accessors add their offset to it.
    /// The transposed `blk * n_evap_hydros + local_idx` stride compiles and silently
    /// aliases one hydro's block onto another's.
    #[inline]
    fn evap_triple_base(&self, local_idx: usize, blk: usize) -> usize {
        self.col_evap_start() + (local_idx * self.n_blks + blk) * EVAP_COLS_PER_HYDRO
    }

    /// Evaporation-outflow column for `(evap hydro local_idx, block blk)` (the
    /// [`EVAP_FLOW_OFFSET`] column of the block's triple).
    #[inline]
    pub(crate) fn evap_flow_col(&self, local_idx: usize, blk: usize) -> usize {
        self.evap_triple_base(local_idx, blk) + EVAP_FLOW_OFFSET
    }

    /// `f_evap_plus` (under-evaporation slack) column for `(evap hydro local_idx,
    /// block blk)` (the [`EVAP_F_PLUS_OFFSET`] column of the block's triple).
    #[inline]
    pub(crate) fn evap_f_plus_col(&self, local_idx: usize, blk: usize) -> usize {
        self.evap_triple_base(local_idx, blk) + EVAP_F_PLUS_OFFSET
    }

    /// `f_evap_minus` (over-evaporation slack) column for `(evap hydro local_idx,
    /// block blk)` (the [`EVAP_F_MINUS_OFFSET`] column of the block's triple).
    #[inline]
    pub(crate) fn evap_f_minus_col(&self, local_idx: usize, blk: usize) -> usize {
        self.evap_triple_base(local_idx, blk) + EVAP_F_MINUS_OFFSET
    }

    /// Deficit column for bus `b_idx`, segment `seg_idx`, block `blk`. Three-term
    /// stride owned by [`BlockGrid::deficit`](crate::indexer::BlockGrid::deficit).
    #[inline]
    pub(crate) fn deficit_col(&self, b_idx: usize, seg_idx: usize, blk: usize) -> usize {
        self.block_grid()
            .deficit(self.col_deficit_start(), b_idx, seg_idx, blk)
    }

    /// Storage column at chronological boundary `k ∈ 0..=K` (`K = self.n_blks`) for
    /// hydro `h`: the single owner of the endpoints-vs-interior split, the storage
    /// analogue of [`Self::block_col`] for flows. The two endpoints are STATE
    /// columns — `k = 0 → S⁰` (incoming state) and `k = K → Sᴷ` (outgoing state) —
    /// while `k ∈ 1..K` are interior CONTROL columns in the `storage_internal`
    /// family (stride `n_blks − 1`, not `n_blks`). The `k == self.n_blks` arm MUST
    /// precede the interior `_` arm, else `_` captures the outgoing endpoint and
    /// addresses an interior column past the family. Outgoing storage returns `h`
    /// because `state.storage.start == 0`, the same convention `resolve_hydro_storage`
    /// uses (`state.storage.start + pos`). Never called in parallel mode (the
    /// single-row balance path is used instead); at `K = 1` only the two endpoints
    /// resolve (no interior).
    #[inline]
    pub(crate) fn block_storage_col(&self, h: usize, k: usize) -> usize {
        match k {
            0 => self.col_storage_in_start() + h,
            k if k == self.n_blks => h,
            _ => self.storage_internal_start + h * (self.n_blks - 1) + (k - 1),
        }
    }

    // ── Role-(a) accessors (read through the borrowed StateLayout handle) ─────────

    /// Theta (future-cost) column; reads `self.state.theta`.
    #[inline]
    #[must_use]
    pub(crate) fn col_theta(&self) -> usize {
        self.state.theta
    }

    /// First incoming-storage column; reads `self.state.storage_in.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_storage_in_start(&self) -> usize {
        self.state.storage_in.start
    }

    /// First AR-lag column; reads `self.state.inflow_lags.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_inflow_lags_start(&self) -> usize {
        self.state.inflow_lags.start
    }

    /// First z-inflow column; reads `self.state.z_inflow.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_z_inflow_start(&self) -> usize {
        self.state.z_inflow.start
    }

    /// First anticipated-state column; reads `self.state.anticipated_state.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_anticipated_state_start(&self) -> usize {
        self.state.anticipated_state.start
    }

    /// Column-side state dimension; reads `self.state.n_state`.
    #[inline]
    #[must_use]
    pub(crate) fn n_state(&self) -> usize {
        self.state.n_state
    }

    // ── Role-(b) accessors (read StageLayout's own fields) ───────────────────────

    /// First turbine-flow column; reads `self.turbine.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_turbine_start(&self) -> usize {
        self.turbine.start
    }

    /// First spillage column; reads `self.spillage.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_spillage_start(&self) -> usize {
        self.spillage.start
    }

    /// First diversion-flow column; reads `self.diversion.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_diversion_start(&self) -> usize {
        self.diversion.start
    }

    /// First thermal column; reads `self.thermal.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_thermal_start(&self) -> usize {
        self.thermal.start
    }

    /// First forward line-flow column; reads `self.line_fwd.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_line_fwd_start(&self) -> usize {
        self.line_fwd.start
    }

    /// First reverse line-flow column; reads `self.line_rev.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_line_rev_start(&self) -> usize {
        self.line_rev.start
    }

    /// First deficit column; reads `self.deficit.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_deficit_start(&self) -> usize {
        self.deficit.start
    }

    /// Maximum deficit segments across buses; reads `self.max_deficit_segments`.
    ///
    /// Test-only: production `deficit_col` routes through
    /// [`BlockGrid::deficit`](crate::indexer::BlockGrid::deficit), which reads
    /// `max_deficit_segments` internally, so the sole live caller is the
    /// deficit-stride regression test that reconstructs the open-coded 3-term
    /// form to prove `deficit_col` still equals it.
    #[cfg(test)]
    #[inline]
    #[must_use]
    pub(crate) fn max_deficit_segments(&self) -> usize {
        self.max_deficit_segments
    }

    /// First excess column; reads `self.excess.start`.
    #[inline]
    #[must_use]
    pub(crate) fn col_excess_start(&self) -> usize {
        self.excess.start
    }

    /// First inflow-slack column; the `inflow_slack` block normalises to `0..0`
    /// without the penalty, so this is the excess-block end cursor — reads
    /// `self.excess.end`.
    #[inline]
    #[must_use]
    pub(crate) fn col_inflow_slack_start(&self) -> usize {
        self.excess.end
    }

    /// First z-inflow definition row; reads `self.z_inflow_row_start`.
    #[inline]
    #[must_use]
    pub(crate) fn row_z_inflow_start(&self) -> usize {
        self.z_inflow_row_start
    }

    /// First water-balance row; reads `self.water_balance.start`.
    #[inline]
    #[must_use]
    pub(crate) fn row_water_balance_start(&self) -> usize {
        self.water_balance.start
    }

    /// First travel-time bucket-definition row; reads `self.transit_bucket_definition.start`.
    #[inline]
    #[must_use]
    pub(crate) fn row_transit_bucket_definition_start(&self) -> usize {
        self.transit_bucket_definition.start
    }

    /// First load-balance row; reads `self.load_balance.start`.
    #[inline]
    #[must_use]
    pub(crate) fn row_load_balance_start(&self) -> usize {
        self.load_balance.start
    }

    /// First FPHA row; the FPHA block follows the load-balance rows, so this is
    /// the load-balance end cursor — reads `self.load_balance.end`.
    #[inline]
    #[must_use]
    pub(crate) fn row_fpha_start(&self) -> usize {
        self.load_balance.end
    }

    // ── Empty-block-cursor accessors (own fields) ────────────────────────────────
    // When a family is empty its range normalises to `0..0`, so each accessor reads
    // a dedicated cursor field (or branches on `self.n_h > 0` to the shared
    // post-equipment cursor) rather than a bare `range.start` that would return `0`
    // and silently misbuild the empty-hydro layout.

    /// Start of FPHA generation columns (one per FPHA hydro per block):
    /// `col_generation_start() + local_fpha_idx * n_blks + blk`.
    #[inline]
    #[must_use]
    pub(crate) fn col_generation_start(&self) -> usize {
        self.generation_col_start
    }

    /// Start of evaporation columns ([`EVAP_COLS_PER_HYDRO`] per `(evap hydro,
    /// block)`, block-major); address the three via `evap_flow_col` /
    /// `evap_f_plus_col` / `evap_f_minus_col`.
    #[inline]
    #[must_use]
    pub(crate) fn col_evap_start(&self) -> usize {
        self.evap_col_start
    }

    /// Start of evaporation constraint rows (one per `(evap hydro, block)`,
    /// block-major): `row_evap_start() + local_evap_idx * n_blks + blk`.
    #[inline]
    #[must_use]
    pub(crate) fn row_evap_start(&self) -> usize {
        self.fpha_rows_end
    }

    /// Start of under-withdrawal slack columns (one per hydro):
    /// `col_withdrawal_neg_start() + h`.
    #[inline]
    #[must_use]
    pub(crate) fn col_withdrawal_neg_start(&self) -> usize {
        if self.n_h > 0 {
            self.withdrawal_slack_neg.start
        } else {
            self.post_equipment_col_start
        }
    }

    /// Start of over-withdrawal slack columns (one per hydro):
    /// `col_withdrawal_pos_start() + h`.
    #[inline]
    #[must_use]
    pub(crate) fn col_withdrawal_pos_start(&self) -> usize {
        if self.n_h > 0 {
            self.withdrawal_slack_pos.start
        } else {
            self.post_equipment_col_start
        }
    }

    /// Start of outflow-below-minimum slack columns (one per hydro per block):
    /// `col_outflow_below_start() + h_idx * n_blks + blk`.
    #[inline]
    #[must_use]
    pub(crate) fn col_outflow_below_start(&self) -> usize {
        if self.n_h > 0 {
            self.outflow_below_slack.start
        } else {
            self.post_equipment_col_start
        }
    }

    /// Start of outflow-above-maximum slack columns (one per hydro per block):
    /// `col_outflow_above_start() + h_idx * n_blks + blk`.
    #[inline]
    #[must_use]
    pub(crate) fn col_outflow_above_start(&self) -> usize {
        if self.n_h > 0 {
            self.outflow_above_slack.start
        } else {
            self.post_equipment_col_start
        }
    }

    /// Start of turbine-below-minimum slack columns (one per hydro per block):
    /// `col_turbine_below_start() + h_idx * n_blks + blk`.
    #[inline]
    #[must_use]
    pub(crate) fn col_turbine_below_start(&self) -> usize {
        if self.n_h > 0 {
            self.turbine_below_slack.start
        } else {
            self.post_equipment_col_start
        }
    }

    /// Start of generation-below-minimum slack columns (one per hydro per block):
    /// `col_generation_below_start() + h_idx * n_blks + blk`.
    #[inline]
    #[must_use]
    pub(crate) fn col_generation_below_start(&self) -> usize {
        if self.n_h > 0 {
            self.generation_below_slack.start
        } else {
            self.post_equipment_col_start
        }
    }

    /// Start of minimum-outflow constraint rows (one per hydro per block):
    /// `row_min_outflow_start() + h_idx * n_blks + blk`.
    #[inline]
    #[must_use]
    pub(crate) fn row_min_outflow_start(&self) -> usize {
        if self.n_h > 0 {
            self.min_outflow_rows.start
        } else {
            self.post_equipment_row_start
        }
    }

    /// Start of maximum-outflow constraint rows (one per hydro per block):
    /// `row_max_outflow_start() + h_idx * n_blks + blk`.
    #[inline]
    #[must_use]
    pub(crate) fn row_max_outflow_start(&self) -> usize {
        if self.n_h > 0 {
            self.max_outflow_rows.start
        } else {
            self.post_equipment_row_start
        }
    }

    /// Start of minimum-turbine constraint rows (one per hydro per block):
    /// `row_min_turbine_start() + h_idx * n_blks + blk`.
    #[inline]
    #[must_use]
    pub(crate) fn row_min_turbine_start(&self) -> usize {
        if self.n_h > 0 {
            self.min_turbine_rows.start
        } else {
            self.post_equipment_row_start
        }
    }

    /// Start of minimum-generation constraint rows (one per hydro per block):
    /// `row_min_generation_start() + h_idx * n_blks + blk`.
    #[inline]
    #[must_use]
    pub(crate) fn row_min_generation_start(&self) -> usize {
        if self.n_h > 0 {
            self.min_generation_rows.start
        } else {
            self.post_equipment_row_start
        }
    }

    /// Start of per-stage `σ_fill`-target rows:
    /// `row_filling_target_start() + local_target_idx` over
    /// `filling_target_hydro_indices`.
    #[inline]
    #[must_use]
    pub(crate) fn row_filling_target_start(&self) -> usize {
        self.row_filling_target_start
    }

    /// Start of per-stage `σ_fill`-target slack columns:
    /// `col_filling_target_start() + local_target_idx`, parallel to
    /// `row_filling_target_start()`.
    #[inline]
    #[must_use]
    pub(crate) fn col_filling_target_start(&self) -> usize {
        self.col_filling_target_start
    }

    /// Start of soft `σ^{v-}` operating-floor rows:
    /// `row_filled_min_storage_floor_start() + local_floor_idx` over
    /// `filled_min_storage_floor_hydro_indices`.
    #[inline]
    #[must_use]
    pub(crate) fn row_filled_min_storage_floor_start(&self) -> usize {
        self.row_filled_min_storage_floor_start
    }

    /// Start of soft `σ^{v-}` operating-floor slack columns:
    /// `col_filled_min_storage_floor_start() + local_floor_idx`, parallel to
    /// `row_filled_min_storage_floor_start()`.
    #[inline]
    #[must_use]
    pub(crate) fn col_filled_min_storage_floor_start(&self) -> usize {
        self.col_filled_min_storage_floor_start
    }
}

#[cfg(test)]
mod tests;
