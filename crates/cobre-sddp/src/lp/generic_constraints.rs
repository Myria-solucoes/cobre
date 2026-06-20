//! Variable reference to LP column index mapping for generic constraints.
//!
//! This module provides `resolve_variable_ref`, which maps a [`VariableRef`]
//! and block index to a list of `(column_index, coefficient_multiplier)` pairs.
//! The LP builder calls this function for each [`cobre_core::LinearTerm`] in a
//! generic constraint expression to produce the CSC matrix entries.
//!
//! ## Column index arithmetic
//!
//! All column offsets follow the layout defined in [`StageIndexer`]; the
//! block-stride arithmetic routes through the [`BlockGrid`] primitive obtained
//! from [`StageIndexer::block_grid`] so the stride expression is single-owned:
//!
//! - Block-level variables (turbine, spillage, thermal, line, excess) use the
//!   flat block-major address [`BlockGrid::flat`] (`start + entity * n_blks + blk`).
//! - Deficit uses the 3-term [`BlockGrid::deficit`] (bus-outer, segment-middle,
//!   block-inner).
//! - Stage-level variables (storage, evaporation, withdrawal) use `col_start + entity_pos`.
//! - FPHA generation uses [`BlockGrid::flat`] on `generation.start`.
//!
//! ## Block expansion
//!
//! When `block_id = None` for a block-level variable, the function returns the
//! column for the *current* `block_idx` rather than expanding to all blocks.
//! The caller iterates over blocks and calls this function once per block, so
//! the per-block expansion happens in the caller loop, not here.
//!
//! ## Pumping columns
//!
//! `PumpingFlow` resolves to the block-major pumping-flow column
//! `grid.flat(col_pumping_start, p_idx, blk)` (via [`BlockGrid::flat`]).
//! `PumpingPower` aliases the SAME
//! flow column scaled by the station's `consumption_mw_per_m3s` rate — power is
//! affine in flow, so it has no column of its own (a separate column would be an
//! unconstrained free variable). The scale matches the bus-balance coupling
//! coefficient.
//!
//! ## Stub entities
//!
//! Variables that reference entity types with no LP columns (contracts,
//! non-controllable sources, withdrawal) return an empty vec. This is consistent
//! with the convention that the constraint term has no LP effect for those
//! entity types.

use std::collections::{BTreeMap, HashMap};

use cobre_core::{CascadeTopology, ConstraintExpression, EntityId, PumpingStation, VariableRef};

use crate::hydro_models::{ProductionModelSet, ResolvedProductionModel};
use crate::indexer::{BlockGrid, StageIndexer};

/// Position maps for entity types, mapping entity IDs to their index in
/// the system's entity arrays.
///
/// Used by [`resolve_variable_ref`] to translate `VariableRef` entity IDs
/// into LP column offsets.
pub(crate) struct EntityPositionMaps<'a> {
    /// Hydro plant ID to position index.
    pub hydro: &'a BTreeMap<EntityId, usize>,
    /// Thermal unit ID to position index.
    pub thermal: &'a BTreeMap<EntityId, usize>,
    /// Bus ID to position index.
    pub bus: &'a BTreeMap<EntityId, usize>,
    /// Line ID to position index.
    pub line: &'a BTreeMap<EntityId, usize>,
}

/// Borrowed cascade context for resolving the total-inflow expression.
///
/// Grouped (rather than passed as two more positional arguments) so
/// [`resolve_variable_ref`] does not trip `clippy::too_many_arguments`,
/// mirroring the [`EntityPositionMaps`] grouping idiom. The `HydroInflow` arm is
/// the only consumer; every other arm ignores it.
pub(crate) struct CascadeRefs<'a> {
    /// Immediately-upstream cascade adjacency. `upstream(h)` returns
    /// `&[EntityId]` sorted by `EntityId.0` at build time, so the resolver
    /// iterates it in a fixed, input-ordering-independent sequence.
    pub cascade: &'a CascadeTopology,
    /// Target hydro id to the **system indices** of plants diverting into it,
    /// built in canonical hydro order (the same representation `matrix.rs`
    /// iterates for the storage-balance diversion-inflow term).
    pub diversion_upstream: &'a HashMap<EntityId, Vec<usize>>,
}

/// Borrowed pumping context for resolving `PumpingFlow`/`PumpingPower`.
///
/// Grouped (rather than passed as more positional arguments) so
/// [`resolve_variable_ref`] does not trip `clippy::too_many_arguments`,
/// mirroring the [`CascadeRefs`] grouping idiom. The `PumpingFlow`/`PumpingPower`
/// arms are the only consumers; every other arm ignores it.
pub(crate) struct PumpingRefs<'a> {
    /// First pumping-flow column. The column for station local index `p_idx`,
    /// block `blk` is `grid.flat(col_pumping_start, p_idx, blk)`, where `grid` is
    /// the [`BlockGrid`](crate::indexer::BlockGrid) sourced from the `StageIndexer`
    /// inside [`resolve_variable_ref`] (the single stride owner every block-major
    /// resolver helper addresses through), not a field here.
    ///
    /// Sourced from `StageLayout::col_pumping_start` (the real reserved range),
    /// **not** from `StageIndexer::pumping_flow` (a permanent `0..0` sentinel).
    /// When `pumping_stations` is empty this value is meaningless, so the resolver
    /// guards on the `pumping_pos` lookup before using it.
    pub col_pumping_start: usize,
    /// Pumping stations in canonical ID-sorted slot order. Indexed by the local
    /// index `p_idx` obtained from [`PumpingRefs::pumping_pos`]; the entry's
    /// `consumption_mw_per_m3s` is the `PumpingPower` coefficient.
    pub pumping_stations: &'a [PumpingStation],
    /// Station id → local index (`p_idx`) into [`PumpingRefs::pumping_stations`].
    /// A lookup miss (unknown station, or no stations at all) yields `vec![]`.
    pub pumping_pos: &'a BTreeMap<EntityId, usize>,
}

/// Map a [`VariableRef`] and block index to LP column indices with multipliers.
///
/// Returns a `Vec<(column_index, coefficient_multiplier)>`. The caller scales
/// each entry by the `LinearTerm::coefficient` to get the final CSC value.
///
/// # Arguments
///
/// - `var_ref` — the LP variable being referenced.
/// - `block_idx` — the block being built (0-indexed). For stage-level variables
///   this is ignored; for block-level variables with `block_id = Some(b)` the
///   function returns the column for block `b` regardless of `block_idx`.
/// - `stage_idx` — stage index used to look up per-stage production models.
/// - `indexer` — column layout for the current stage LP.
/// - `production_models` — resolved production model set, used to distinguish
///   FPHA hydros from constant-productivity hydros for `HydroGeneration`.
/// - `positions` — entity position maps grouped into [`EntityPositionMaps`].
/// - `cascade_refs` — cascade topology + diversion-into map grouped into
///   [`CascadeRefs`]; consulted only by the `HydroInflow` arm.
/// - `pumping_refs` — pumping column start, station slice, and position map
///   grouped into [`PumpingRefs`]; consulted only by the `PumpingFlow` /
///   `PumpingPower` arms.
///
/// # Returns
///
/// An empty vec when:
/// - The entity ID is not found in the relevant position map (should have been
///   caught by referential validation, but this is defense-in-depth).
/// - The variable type references a stub entity with no LP columns (contracts,
///   non-controllable sources, withdrawal).
#[must_use]
// Rationale: one exhaustive arm per `VariableRef` variant — the match is the
// dispatch, and its exhaustiveness is the contract that a new variant forces a
// compile error here. Splitting it into sub-dispatchers to satisfy the
// line-count heuristic would fragment that closed-set guarantee for no gain.
#[allow(clippy::too_many_lines)]
pub(crate) fn resolve_variable_ref(
    var_ref: &VariableRef,
    block_idx: usize,
    stage_idx: usize,
    indexer: &StageIndexer,
    production_models: &ProductionModelSet,
    positions: &EntityPositionMaps<'_>,
    cascade_refs: &CascadeRefs<'_>,
    pumping_refs: &PumpingRefs<'_>,
) -> Vec<(usize, f64)> {
    let hydro_pos = positions.hydro;
    let thermal_pos = positions.thermal;
    let bus_pos = positions.bus;
    let line_pos = positions.line;
    let grid = indexer.block_grid();
    match var_ref {
        // ── Stage-level hydro variables ────────────────────────────────────
        VariableRef::HydroStorage { hydro_id } => {
            resolve_hydro_storage(*hydro_id, indexer, hydro_pos)
        }

        VariableRef::HydroEvaporation { hydro_id } => {
            resolve_hydro_evaporation(*hydro_id, indexer, hydro_pos)
        }

        // ── Block-level hydro variables ────────────────────────────────────
        VariableRef::HydroInflow { hydro_id, block_id } => resolve_hydro_inflow(
            *hydro_id,
            *block_id,
            block_idx,
            grid,
            indexer,
            hydro_pos,
            cascade_refs,
        ),

        VariableRef::HydroTurbined { hydro_id, block_id } => resolve_block_variable(
            *hydro_id,
            *block_id,
            block_idx,
            grid,
            block_col_range(indexer, ElementKind::Turbine).start,
            hydro_pos,
            1.0,
        ),

        VariableRef::HydroSpillage { hydro_id, block_id } => resolve_block_variable(
            *hydro_id,
            *block_id,
            block_idx,
            grid,
            block_col_range(indexer, ElementKind::Spillage).start,
            hydro_pos,
            1.0,
        ),

        VariableRef::HydroOutflow { hydro_id, block_id } => {
            resolve_hydro_outflow(*hydro_id, *block_id, block_idx, grid, indexer, hydro_pos)
        }

        VariableRef::HydroGeneration { hydro_id, block_id } => resolve_hydro_generation(
            *hydro_id,
            *block_id,
            block_idx,
            grid,
            stage_idx,
            indexer,
            production_models,
            hydro_pos,
        ),

        // ── Thermal ────────────────────────────────────────────────────────
        VariableRef::ThermalGeneration {
            thermal_id,
            block_id,
        } => resolve_block_variable(
            *thermal_id,
            *block_id,
            block_idx,
            grid,
            block_col_range(indexer, ElementKind::Thermal).start,
            thermal_pos,
            1.0,
        ),

        // ── Transmission lines ─────────────────────────────────────────────
        VariableRef::LineDirect { line_id, block_id } => resolve_block_variable(
            *line_id,
            *block_id,
            block_idx,
            grid,
            block_col_range(indexer, ElementKind::LineFwd).start,
            line_pos,
            1.0,
        ),

        VariableRef::LineReverse { line_id, block_id } => resolve_block_variable(
            *line_id,
            *block_id,
            block_idx,
            grid,
            block_col_range(indexer, ElementKind::LineRev).start,
            line_pos,
            1.0,
        ),

        VariableRef::LineExchange { line_id, block_id } => {
            resolve_line_exchange(*line_id, *block_id, block_idx, grid, indexer, line_pos)
        }

        // ── Bus deficit / excess ───────────────────────────────────────────
        VariableRef::BusDeficit { bus_id, block_id } => {
            resolve_bus_deficit(*bus_id, *block_id, block_idx, grid, indexer, bus_pos)
        }

        VariableRef::BusExcess { bus_id, block_id } => resolve_block_variable(
            *bus_id,
            *block_id,
            block_idx,
            grid,
            block_col_range(indexer, ElementKind::Excess).start,
            bus_pos,
            1.0,
        ),

        VariableRef::HydroDiversion { hydro_id, block_id } => resolve_block_variable(
            *hydro_id,
            *block_id,
            block_idx,
            grid,
            block_col_range(indexer, ElementKind::Diversion).start,
            hydro_pos,
            1.0,
        ),

        // ── Anticipated thermal decision column ────────────────────────────
        VariableRef::AnticipatedDecision { thermal_id } => {
            resolve_anticipated_decision(*thermal_id, indexer, thermal_pos)
        }

        // ── Pumping columns ────────────────────────────────────────────────
        // PumpingFlow carries +1.0; PumpingPower aliases the SAME flow column
        // scaled by the station's consumption rate (power is affine in flow),
        // so the coefficient is computed from `pumping_stations[p_idx]`.
        VariableRef::PumpingFlow {
            station_id,
            block_id,
        } => resolve_pumping_column(
            *station_id,
            *block_id,
            block_idx,
            grid,
            pumping_refs,
            |_| 1.0,
        ),

        VariableRef::PumpingPower {
            station_id,
            block_id,
        } => resolve_pumping_column(
            *station_id,
            *block_id,
            block_idx,
            grid,
            pumping_refs,
            |station| station.consumption_mw_per_m3s,
        ),

        // ── Contracts ──────────────────────────────────────────────────────
        // Contracts carry no LP decision column in this implementation: the
        // `ElementKind::ContractImport`/`ContractExport` families resolve through
        // `block_col_range` (their single owner) to the empty `0..0` range, so a
        // generic-constraint term referencing a contract emits no
        // `(column, coefficient)` pair. The `debug_assert!` is the seam a future
        // contract-column implementation extends: it both keeps the `ElementKind`
        // variants and `block_col_range`'s contract arm live and fires loudly if a
        // non-empty range is ever wired without also emitting the resolved
        // column(s) here — preventing a silent fall-through to the empty return.
        VariableRef::ContractImport { .. } => {
            debug_assert!(
                block_col_range(indexer, ElementKind::ContractImport).is_empty(),
                "ContractImport gained an LP column range but this arm still emits no \
                 (column, coefficient) pair — wire the contract-column resolution here",
            );
            vec![]
        }
        VariableRef::ContractExport { .. } => {
            debug_assert!(
                block_col_range(indexer, ElementKind::ContractExport).is_empty(),
                "ContractExport gained an LP column range but this arm still emits no \
                 (column, coefficient) pair — wire the contract-column resolution here",
            );
            vec![]
        }

        // ── Stub entities with no LP columns ──────────────────────────────
        // These entity types are registered in the data model but contribute no LP
        // decision variable, and have no `ElementKind` block-major column family:
        // - HydroWithdrawal: withdrawal is a schedule fixed by bounds, not a
        //   decision variable.
        // - NonControllableGeneration, NonControllableCurtailment: non-controllable
        //   sources carry no decision column.
        VariableRef::HydroWithdrawal { .. }
        | VariableRef::NonControllableGeneration { .. }
        | VariableRef::NonControllableCurtailment { .. } => vec![],
    }
}

/// Whether a single [`VariableRef`] resolves to the *same* LP column(s)
/// regardless of the block index passed to [`resolve_variable_ref`].
///
/// A variable is **block-independent** ("stock") when its resolver ignores the
/// `block_idx` argument and returns a per-stage column. Only three kinds qualify,
/// each verified against its resolver:
///
/// - [`VariableRef::HydroStorage`] → `storage.start + pos` (outgoing reservoir level)
/// - [`VariableRef::HydroEvaporation`] → the stage-level evaporation-outflow column
/// - [`VariableRef::AnticipatedDecision`] → `anticipated_decision.start + local`
///   (a per-plant per-stage scalar, uniform across blocks)
///
/// [`VariableRef::HydroInflow`] is **block-dependent** (in the `false` arm): its
/// upstream-release terms (`turbine`/`spillage`/`diversion` of upstream plants)
/// are per-block LP columns, and a single block-dependent term makes the whole
/// expression block-dependent. Classifying it "stock" would collapse a multi-block
/// expression to one mis-priced stage-level row that reads upstream columns at a
/// single arbitrary block, silently dropping the other blocks' contributions.
///
/// Every other kind is block-level: it resolves through the block-stride
/// [`BlockGrid`] primitive (via `resolve_block_variable` / `resolve_hydro_inflow`
/// / FPHA generation / line / deficit / excess / pumping), so distinct block
/// indices yield distinct columns. [`VariableRef::PumpingFlow`] and
/// [`VariableRef::PumpingPower`] are block-level (their `resolve_pumping_column`
/// column is a [`BlockGrid::flat`] address on `col_pumping_start`), so they stay
/// in the `false` arm — classifying them "stock" would collapse a per-block
/// pumping term to a single mis-priced row. The remaining stub kinds (withdrawal, contracts,
/// non-controllable) resolve to no columns at all; they are conservatively treated
/// as block-level here so that only *provably* stock variables enable the
/// single-row collapse.
///
/// The match is deliberately exhaustive (no wildcard arm): a future
/// [`VariableRef`] variant forces a compile error here, defaulting nothing to
/// "stock" by omission.
#[must_use]
fn variable_ref_is_block_independent(var_ref: &VariableRef) -> bool {
    match var_ref {
        VariableRef::HydroStorage { .. }
        | VariableRef::HydroEvaporation { .. }
        | VariableRef::AnticipatedDecision { .. } => true,
        VariableRef::HydroInflow { .. }
        | VariableRef::HydroTurbined { .. }
        | VariableRef::HydroSpillage { .. }
        | VariableRef::HydroDiversion { .. }
        | VariableRef::HydroOutflow { .. }
        | VariableRef::HydroGeneration { .. }
        | VariableRef::ThermalGeneration { .. }
        | VariableRef::LineDirect { .. }
        | VariableRef::LineReverse { .. }
        | VariableRef::LineExchange { .. }
        | VariableRef::BusDeficit { .. }
        | VariableRef::BusExcess { .. }
        | VariableRef::HydroWithdrawal { .. }
        | VariableRef::PumpingFlow { .. }
        | VariableRef::PumpingPower { .. }
        | VariableRef::ContractImport { .. }
        | VariableRef::ContractExport { .. }
        | VariableRef::NonControllableGeneration { .. }
        | VariableRef::NonControllableCurtailment { .. } => false,
    }
}

/// Whether an entire generic-constraint expression is block-independent.
///
/// True only when **every** term is block-independent (see
/// [`variable_ref_is_block_independent`]). A `block_id = None` bound on such an
/// expression produces the *same* row for every block, so the LP builder may
/// collapse the per-block replication into a single stage-level row priced by
/// the stage's total hours — the row's coefficients do not depend on the block.
///
/// An **empty** expression (no terms) is vacuously block-independent: it carries
/// no block-dependent coefficients, so collapsing its replicated rows changes
/// nothing but the row/slack count.
///
/// Any block-level term anywhere forces `false` (keep the per-block path).
#[must_use]
pub(crate) fn expression_is_block_independent(expression: &ConstraintExpression) -> bool {
    expression
        .terms
        .iter()
        .all(|term| variable_ref_is_block_independent(&term.variable))
}

/// Resolve `HydroStorage` to its stage-level outgoing storage column.
///
/// `indexer.storage[h] = storage.start + h`. Returns empty vec when the hydro
/// ID is not found in `hydro_pos`.
fn resolve_hydro_storage(
    hydro_id: EntityId,
    indexer: &StageIndexer,
    hydro_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    if let Some(&pos) = hydro_pos.get(&hydro_id) {
        vec![(indexer.storage.start + pos, 1.0)]
    } else {
        vec![]
    }
}

/// Resolve `HydroInflow` to the cascade total-inflow expression at `eff_blk`.
///
/// This is the inflow side of the hydro water balance expressed as an
/// instantaneous **rate** (m³/s): the incremental (local) `z_inflow` column plus
/// the immediately-upstream cascade releases (turbine + spillage) plus the
/// plants diverting into `h`. All coefficients are unit `+1.0` — a rate identity,
/// **not** the `−τ`-weighted (hm³) storage-balance row. The `−τ` sign and `τ`
/// weighting are specific to the storage-balance row and must not be copied here.
/// `h`'s own outflows (turbine/spillage/diversion), evaporation, withdrawal
/// slacks, AR-lag-`ψ`, and pumped transfer are **excluded**: they are
/// storage-balance (`±τ`-weighted hm³) / loss / outflow terms or have no LP
/// column, not instantaneous-rate inflow terms.
///
/// Column arithmetic mirrors [`resolve_block_variable`]: the flat block-major
/// address `grid.flat(col_start, pos, eff_blk)` with
/// `eff_blk = block_id.unwrap_or(block_idx)`.
/// The column set mirrors the cascade/diversion loops of
/// `matrix.rs::fill_state_and_water_entries`: upstream releases iterate
/// `cascade.upstream(h)` resolved via `hydro_pos`; diverted inflow iterates
/// `diversion_upstream[h]`, whose values are already **system indices**. Both are
/// in canonical (ID-sorted / canonical-hydro) order at build time, so the emitted
/// pairs are input-ordering-independent with no extra sort here.
///
/// The `z_inflow.is_empty()` guard is load-bearing: `z_inflow` is empty when
/// `hydro_count == 0` (unlike `storage`, which is non-empty whenever hydros
/// exist), so `z_inflow.start` would be a meaningless offset there. Returns an
/// empty vec when `hydro_count == 0` or `hydro_id` is unknown.
fn resolve_hydro_inflow(
    hydro_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    indexer: &StageIndexer,
    hydro_pos: &BTreeMap<EntityId, usize>,
    cascade_refs: &CascadeRefs<'_>,
) -> Vec<(usize, f64)> {
    if indexer.z_inflow.is_empty() {
        return vec![];
    }
    let Some(&pos_h) = hydro_pos.get(&hydro_id) else {
        return vec![];
    };

    let eff_blk = block_id.unwrap_or(block_idx);
    let upstream = cascade_refs.cascade.upstream(hydro_id);
    let diversion_into = cascade_refs
        .diversion_upstream
        .get(&hydro_id)
        .map_or(&[][..], Vec::as_slice);

    // Incremental term, then two per upstream plant, then one per diverting plant.
    let mut result = Vec::with_capacity(1 + 2 * upstream.len() + diversion_into.len());

    // Incremental (local) inflow: the free z_inflow column.
    result.push((indexer.z_inflow.start + pos_h, 1.0));

    // Upstream cascade releases: turbine + spillage of each immediately-upstream
    // plant at the effective block. Same column set as the storage-balance inflow
    // side, but with +1.0 (rate) instead of −τ (volume).
    let turbine = block_col_range(indexer, ElementKind::Turbine);
    let spillage = block_col_range(indexer, ElementKind::Spillage);
    if !turbine.is_empty() && !spillage.is_empty() {
        for &up_id in upstream {
            if let Some(&pos_up) = hydro_pos.get(&up_id) {
                result.push((grid.flat(turbine.start, pos_up, eff_blk), 1.0));
                result.push((grid.flat(spillage.start, pos_up, eff_blk), 1.0));
            }
        }
    }

    // Diverted inflow: the diversion column of each plant diverting into `h`.
    // `diversion_upstream[h]` already holds system indices, so no `hydro_pos`
    // lookup is needed (mirrors the matrix.rs diversion-inflow loop).
    let diversion = block_col_range(indexer, ElementKind::Diversion);
    if !diversion.is_empty() {
        for &d_idx in diversion_into {
            result.push((grid.flat(diversion.start, d_idx, eff_blk), 1.0));
        }
    }

    result
}

/// Resolve `HydroEvaporation` to the evaporation-outflow column for the matching hydro.
///
/// The evaporation list uses a local index; we find it by matching the
/// system-level hydro position. Returns empty vec when the hydro has no
/// linearized evaporation at this stage.
fn resolve_hydro_evaporation(
    hydro_id: EntityId,
    indexer: &StageIndexer,
    hydro_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    let Some(&sys_pos) = hydro_pos.get(&hydro_id) else {
        return vec![];
    };
    // Linear scan over the small per-stage evaporation-hydro list. This runs
    // on the cold per-constraint-term resolver path (template build, not a
    // solve loop), so the O(n) cost over a handful of evaporation hydros is
    // not measurable and a pre-built O(1) reverse map is not warranted here —
    // unlike `resolve_anticipated_decision`, whose larger/hotter set earns
    // `anticipated_local_by_sys_pos`.
    let Some(local_idx) = indexer
        .evap_hydro_indices
        .iter()
        .position(|&p| p == sys_pos)
    else {
        return vec![];
    };
    let evaporation_flow_col = indexer.evap_indices[local_idx].evaporation_flow_col;
    vec![(evaporation_flow_col, 1.0)]
}

/// Resolve `HydroOutflow` (turbine + spillage) to two block-level columns.
///
/// One `hydro_pos` lookup resolves both columns (mirroring
/// [`resolve_line_exchange`]); turbine is emitted before spillage. Returns an
/// empty vec on a `hydro_pos` miss, so the no-result-on-miss contract holds for
/// the whole pair, never a partial single column.
fn resolve_hydro_outflow(
    hydro_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    indexer: &StageIndexer,
    hydro_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    let Some(&pos) = hydro_pos.get(&hydro_id) else {
        return vec![];
    };
    let effective_blk = block_id.unwrap_or(block_idx);
    let turbine_col = grid.flat(
        block_col_range(indexer, ElementKind::Turbine).start,
        pos,
        effective_blk,
    );
    let spillage_col = grid.flat(
        block_col_range(indexer, ElementKind::Spillage).start,
        pos,
        effective_blk,
    );
    vec![(turbine_col, 1.0), (spillage_col, 1.0)]
}

/// Resolve `HydroGeneration` by dispatching on the production model.
///
/// - FPHA hydros: maps to the generation column at
///   `grid.flat(generation.start, fpha_local_idx, blk)` (via [`BlockGrid::flat`]).
/// - Constant-productivity hydros: maps to the turbine column scaled by productivity.
fn resolve_hydro_generation(
    hydro_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    stage_idx: usize,
    indexer: &StageIndexer,
    production_models: &ProductionModelSet,
    hydro_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    let Some(&sys_pos) = hydro_pos.get(&hydro_id) else {
        return vec![];
    };
    match production_models.model(sys_pos, stage_idx) {
        ResolvedProductionModel::Fpha { .. } => {
            // Linear scan over the small per-stage FPHA-hydro list. This runs on
            // the cold per-constraint-term resolver path (template build, not a
            // solve loop), so the O(n) cost over a handful of FPHA hydros is not
            // measurable and a pre-built O(1) reverse map is not warranted here —
            // unlike `resolve_anticipated_decision`, whose larger/hotter set earns
            // `anticipated_local_by_sys_pos`.
            if let Some(fpha_local_idx) = indexer
                .fpha_hydro_indices
                .iter()
                .position(|&p| p == sys_pos)
            {
                let effective_blk = block_id.unwrap_or(block_idx);
                let col = grid.flat(indexer.generation.start, fpha_local_idx, effective_blk);
                vec![(col, 1.0)]
            } else {
                // Should not happen if indexer and production_models are consistent.
                vec![]
            }
        }
        ResolvedProductionModel::ConstantProductivity { productivity } => {
            // generation = productivity * turbined → map to turbine column.
            resolve_block_variable(
                hydro_id,
                block_id,
                block_idx,
                grid,
                block_col_range(indexer, ElementKind::Turbine).start,
                hydro_pos,
                *productivity,
            )
        }
    }
}

/// Resolve `LineExchange` (net = forward − reverse) to two columns with signs.
fn resolve_line_exchange(
    line_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    indexer: &StageIndexer,
    line_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    if let Some(&pos) = line_pos.get(&line_id) {
        let effective_blk = block_id.unwrap_or(block_idx);
        let fwd_col = grid.flat(
            block_col_range(indexer, ElementKind::LineFwd).start,
            pos,
            effective_blk,
        );
        let rev_col = grid.flat(
            block_col_range(indexer, ElementKind::LineRev).start,
            pos,
            effective_blk,
        );
        vec![(fwd_col, 1.0), (rev_col, -1.0)]
    } else {
        vec![]
    }
}

/// Resolve `BusDeficit` to one column per deficit segment.
///
/// Each segment's column is the deficit 3-term address
/// `grid.deficit(deficit.start, b_pos, seg, blk)` (bus-outer, segment-middle,
/// block-inner); see [`BlockGrid::deficit`]. The segment count `S` comes from
/// `indexer.max_deficit_segments` because the grid exposes no accessor for it.
fn resolve_bus_deficit(
    bus_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    indexer: &StageIndexer,
    bus_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    if let Some(&b_pos) = bus_pos.get(&bus_id) {
        let effective_blk = block_id.unwrap_or(block_idx);
        let s = indexer.max_deficit_segments;
        (0..s)
            .map(|seg| {
                (
                    grid.deficit(indexer.deficit.start, b_pos, seg, effective_blk),
                    1.0,
                )
            })
            .collect()
    } else {
        vec![]
    }
}

/// Resolve `AnticipatedDecision` to the per-plant stage-level decision column.
///
/// Column layout: `anticipated_decision.start + local_idx` where `local_idx`
/// is the position of the thermal's system index in
/// `indexer.anticipated_thermal_indices`.
///
/// Returns an empty vec when:
/// - `thermal_id` is not in `thermal_pos` (defense-in-depth; referential
///   validation should have caught this).
/// - The thermal's system position is not in `anticipated_thermal_indices`
///   (the thermal is not anticipated; semantic validation should have caught
///   this via rule 17 in `check_anticipated_decision_target_is_anticipated`).
fn resolve_anticipated_decision(
    thermal_id: EntityId,
    indexer: &StageIndexer,
    thermal_pos: &BTreeMap<EntityId, usize>,
) -> Vec<(usize, f64)> {
    let Some(&sys_pos) = thermal_pos.get(&thermal_id) else {
        return vec![];
    };
    // O(1) reverse lookup via the pre-built map in the indexer rather than
    // a linear scan over anticipated_thermal_indices.
    if let Some(&local_idx) = indexer.anticipated_local_by_sys_pos.get(&sys_pos) {
        vec![(indexer.anticipated_decision.start + local_idx, 1.0)]
    } else {
        vec![]
    }
}

/// Resolve `PumpingFlow`/`PumpingPower` to the block-major pumping-flow column(s).
///
/// Both variants resolve to the SAME flow column — `PumpingPower` has no column
/// of its own; it aliases the flow column scaled by the station's
/// `consumption_mw_per_m3s` (power is affine in flow). Resolving `PumpingPower`
/// to a separate column would create an unconstrained free variable. The
/// coefficient is therefore selected by `coeff_fn`: `PumpingFlow` passes `|_| 1.0`,
/// `PumpingPower` passes `|s| s.consumption_mw_per_m3s` — the same alias as the
/// bus-balance coupling coefficient.
///
/// Column arithmetic: the flat block-major address
/// `grid.flat(col_pumping_start, p_idx, eff_blk)` where
/// `eff_blk = block_id.unwrap_or(block_idx)`. For `block_id = None` the caller
/// loop supplies the effective block via `block_idx` (one resolver call per
/// block), mirroring [`resolve_block_variable`]; this function returns a single
/// pair per call regardless of `block_id`.
///
/// Returns an empty vec when the station id is unknown (a `pumping_pos` miss) or
/// when there are no pumping stations (`pumping_pos` is empty), so `n_pumping == 0`
/// is handled by the same lookup guard. No panic.
fn resolve_pumping_column(
    station_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    grid: BlockGrid,
    pumping_refs: &PumpingRefs<'_>,
    coeff_fn: impl Fn(&PumpingStation) -> f64,
) -> Vec<(usize, f64)> {
    let Some(&p_idx) = pumping_refs.pumping_pos.get(&station_id) else {
        return vec![];
    };
    // Defense-in-depth: `pumping_pos` and `pumping_stations` are built from the
    // same ID-sorted slice, so a present index is always in range; guard rather
    // than index to uphold the no-panic contract if they ever diverge.
    let Some(station) = pumping_refs.pumping_stations.get(p_idx) else {
        return vec![];
    };
    let eff_blk = block_id.unwrap_or(block_idx);
    let col = grid.flat(pumping_refs.col_pumping_start, p_idx, eff_blk);
    vec![(col, coeff_fn(station))]
}

/// Resolve a block-level LP variable to a `(column_index, multiplier)` pair.
///
/// Computes the flat block-major address `grid.flat(col_start, entity_pos,
/// effective_block_idx)` where `effective_block_idx` is `block_idx` when
/// `ref_block_id` is `None`, or `b` when `ref_block_id` is `Some(b)`. Routing
/// through [`BlockGrid::flat`] keeps the flat stride expression single-owned —
/// this resolver and `StageLayout::block_col` no longer open-code it
/// independently.
///
/// Returns an empty vec if the entity ID is not found in `pos_map`.
fn resolve_block_variable(
    entity_id: EntityId,
    ref_block_id: Option<usize>,
    current_block_idx: usize,
    grid: BlockGrid,
    col_start: usize,
    pos_map: &BTreeMap<EntityId, usize>,
    multiplier: f64,
) -> Vec<(usize, f64)> {
    if let Some(&pos) = pos_map.get(&entity_id) {
        let effective_blk = ref_block_id.unwrap_or(current_block_idx);
        vec![(grid.flat(col_start, pos, effective_blk), multiplier)]
    } else {
        vec![]
    }
}

/// A block-major equipment/line/contract column family the resolver addresses by
/// reading the family's [`StageIndexer`]-owned column range. Each variant maps to
/// exactly one range in [`block_col_range`].
///
/// Closed set, exhaustively matched (no `_` arm) in [`block_col_range`]: a new
/// block-major family is a compile error there until its range source is named,
/// rather than silently resolving to whichever field a hand-written `.start` read
/// happened to pick.
#[derive(Clone, Copy)]
enum ElementKind {
    Turbine,
    Spillage,
    Diversion,
    Thermal,
    LineFwd,
    LineRev,
    Excess,
    ContractImport,
    ContractExport,
}

/// Map an [`ElementKind`] to its block-major column range on `indexer`.
///
/// This is the single point that pairs an element family with its [`StageIndexer`]
/// column range, so a wrong arm mapping (e.g. returning `indexer.spillage` for
/// `Turbine`) is caught here, once, instead of being open-coded — and silently
/// wrong — at each `col_start` read across the resolver and its helpers.
///
/// `ContractImport`/`ContractExport` return the empty `0..0` range: contracts have
/// no LP columns in this implementation (the indexer exposes no contract range), so
/// they resolve to no `(column, coefficient)` pair. This is the single owner of the
/// contract column range, so wiring real contract columns later is one edit here.
///
/// Returns an **owned** `Range<usize>` (a two-`usize` clone, stack-only) so the
/// `0..0` contract sentinel is returnable without tying the result's lifetime to
/// `indexer`; do NOT change this to `&Range<usize>`.
#[must_use]
fn block_col_range(indexer: &StageIndexer, kind: ElementKind) -> std::ops::Range<usize> {
    match kind {
        ElementKind::Turbine => indexer.turbine.clone(),
        ElementKind::Spillage => indexer.spillage.clone(),
        ElementKind::Diversion => indexer.diversion.clone(),
        ElementKind::Thermal => indexer.thermal.clone(),
        ElementKind::LineFwd => indexer.line_fwd.clone(),
        ElementKind::LineRev => indexer.line_rev.clone(),
        ElementKind::Excess => indexer.excess.clone(),
        ElementKind::ContractImport | ElementKind::ContractExport => 0..0,
    }
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::identity_op,
    clippy::erasing_op
)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use cobre_core::entities::{HydroGenerationModel, HydroPenalties};
    use cobre_core::{CascadeTopology, EntityId, Hydro, PumpingStation, VariableRef};

    use super::{
        CascadeRefs, ElementKind, PumpingRefs, block_col_range, resolve_variable_ref,
        variable_ref_is_block_independent,
    };
    use crate::hydro_models::{FphaPlane, ProductionModelSet, ResolvedProductionModel};
    use crate::indexer::StageIndexer;

    // ── Test helpers ──────────────────────────────────────────────────────────

    /// Minimal `Hydro` carrying only the `id`/`downstream_id` that
    /// [`CascadeTopology::build`] reads; every other field is an inert default.
    /// Mirrors the `make_hydro` helper in `cobre-core`'s cascade tests.
    fn make_hydro(id: i32, downstream_id: Option<i32>) -> Hydro {
        let zero_penalties = HydroPenalties {
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
            inflow_nonnegativity_cost: 1000.0,
        };
        Hydro {
            id: EntityId(id),
            name: String::new(),
            bus_id: EntityId(0),
            downstream_id: downstream_id.map(EntityId),
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties,
        }
    }

    /// An empty cascade (no upstream links) for the resolver paths that ignore it.
    fn empty_cascade() -> CascadeTopology {
        CascadeTopology::build(&[])
    }

    /// Build a `StageIndexer` with equipment for tests.
    ///
    /// N=4 hydros (2 FPHA at positions 0, 2), L=0, T=2 thermals, Ln=1 line, B=2 buses, K=3 blocks.
    /// S=2 max deficit segments.
    ///
    /// Column layout:
    ///   storage:   [0, 4)         = 0..4
    ///   lags:      [4, 4*(1+0))   = 4..4   (L=0, empty)
    ///   z_inflow:  [4*(1+0), 4*(2+0)) = 4..8
    ///   storage_in:[4*(2+0), 4*(3+0)) = 8..12
    ///   theta = N*(3+L) = 4*(3+0) = 12
    ///   decision_start = 13
    ///   turbine:    [13, 13+4*3) = 13..25   (4 hydros * 3 blocks)
    ///   spillage:   [25, 25+4*3) = 25..37
    ///   diversion:  [37, 37+4*3) = 37..49  (4 hydros * 3 blocks)
    ///   thermal:    [49, 49+2*3) = 49..55  (2 thermals * 3 blocks)
    ///   line_fwd:   [55, 55+1*3) = 55..58  (1 line * 3 blocks)
    ///   line_rev:   [58, 58+1*3) = 58..61
    ///   deficit:    [61, 61+2*2*3) = 61..73 (2 buses * 2 segs * 3 blocks)
    ///   excess:     [73, 73+2*3) = 73..79  (2 buses * 3 blocks)
    ///   generation: [79, 79+2*3) = 79..85  (2 FPHA hydros * 3 blocks)
    ///   evap: none
    ///   withdrawal_slack_neg: [85, 89)  withdrawal_slack_pos: [89, 93) (4 hydros)
    ///
    /// Storage: 0..4
    fn make_indexer() -> StageIndexer {
        // N=4, L=0, T=2, Ln=1, B=2, K=3, no penalty, 2 FPHA hydros at positions 0 and 2
        // (local FPHA indices 0 and 1), each with 3 planes.
        StageIndexer::with_equipment_and_evaporation(
            &crate::indexer::EquipmentCounts {
                hydro_count: 4,
                max_par_order: 0,
                n_thermals: 2,
                n_lines: 1,
                n_buses: 2,
                n_blks: 3,
                has_inflow_penalty: false,
                max_deficit_segments: 2,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                n_pumping: 0,
            },
            &crate::indexer::FphaColumnLayout {
                hydro_indices: vec![0, 2],
                planes_per_hydro: vec![3, 3],
            },
            &crate::indexer::EvapConfig {
                hydro_indices: vec![],
            },
        )
    }

    /// Build a `ProductionModelSet` for 4 hydros and 2 stages.
    ///
    /// - Hydro 0: FPHA at all stages
    /// - Hydro 1: ConstantProductivity(2.5) at all stages
    /// - Hydro 2: FPHA at all stages
    /// - Hydro 3: ConstantProductivity(1.0) at all stages
    fn make_production_models() -> ProductionModelSet {
        let fpha_plane = FphaPlane {
            intercept: 0.0,
            gamma_v: 0.1,
            gamma_q: 0.5,
            gamma_s: 0.0,
        };
        let fpha_model = || ResolvedProductionModel::Fpha {
            planes: vec![fpha_plane],
        };
        let models: Vec<Vec<ResolvedProductionModel>> = vec![
            vec![fpha_model(), fpha_model()], // hydro 0 — FPHA
            vec![
                ResolvedProductionModel::ConstantProductivity { productivity: 2.5 },
                ResolvedProductionModel::ConstantProductivity { productivity: 2.5 },
            ], // hydro 1 — constant
            vec![fpha_model(), fpha_model()], // hydro 2 — FPHA
            vec![
                ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
                ResolvedProductionModel::ConstantProductivity { productivity: 1.0 },
            ], // hydro 3 — constant
        ];
        ProductionModelSet::new(models, 4, 2)
    }

    fn make_hydro_pos() -> BTreeMap<EntityId, usize> {
        // Hydros with EntityId 10, 20, 30, 40 at system positions 0, 1, 2, 3
        [
            (EntityId(10), 0),
            (EntityId(20), 1),
            (EntityId(30), 2),
            (EntityId(40), 3),
        ]
        .into_iter()
        .collect()
    }

    fn make_thermal_pos() -> BTreeMap<EntityId, usize> {
        // Thermals with EntityId 5 and 6 at positions 0 and 1
        [(EntityId(5), 0), (EntityId(6), 1)].into_iter().collect()
    }

    fn make_bus_pos() -> BTreeMap<EntityId, usize> {
        // Buses with EntityId 100, 200 at positions 0, 1
        [(EntityId(100), 0), (EntityId(200), 1)]
            .into_iter()
            .collect()
    }

    fn make_line_pos() -> BTreeMap<EntityId, usize> {
        // Line with EntityId 50 at position 0
        [(EntityId(50), 0)].into_iter().collect()
    }

    fn call(
        var_ref: VariableRef,
        block_idx: usize,
        indexer: &StageIndexer,
        production_models: &ProductionModelSet,
        hydro_pos: &BTreeMap<EntityId, usize>,
        thermal_pos: &BTreeMap<EntityId, usize>,
        bus_pos: &BTreeMap<EntityId, usize>,
        line_pos: &BTreeMap<EntityId, usize>,
    ) -> Vec<(usize, f64)> {
        // Paths under test here ignore the cascade context; pass an empty one.
        let cascade = empty_cascade();
        let diversion_upstream: HashMap<EntityId, Vec<usize>> = HashMap::new();
        call_with_cascade(
            var_ref,
            block_idx,
            indexer,
            production_models,
            hydro_pos,
            thermal_pos,
            bus_pos,
            line_pos,
            &cascade,
            &diversion_upstream,
        )
    }

    /// Like [`call`], but threads an explicit cascade topology and
    /// diversion-into map for the `HydroInflow` total-inflow tests.
    fn call_with_cascade(
        var_ref: VariableRef,
        block_idx: usize,
        indexer: &StageIndexer,
        production_models: &ProductionModelSet,
        hydro_pos: &BTreeMap<EntityId, usize>,
        thermal_pos: &BTreeMap<EntityId, usize>,
        bus_pos: &BTreeMap<EntityId, usize>,
        line_pos: &BTreeMap<EntityId, usize>,
        cascade: &CascadeTopology,
        diversion_upstream: &HashMap<EntityId, Vec<usize>>,
    ) -> Vec<(usize, f64)> {
        let positions = super::EntityPositionMaps {
            hydro: hydro_pos,
            thermal: thermal_pos,
            bus: bus_pos,
            line: line_pos,
        };
        let cascade_refs = CascadeRefs {
            cascade,
            diversion_upstream,
        };
        // Non-pumping paths ignore the pumping context; pass an empty one (no
        // stations), so the PumpingFlow/PumpingPower lookup misses and yields [].
        let no_stations: Vec<PumpingStation> = Vec::new();
        let empty_pumping_pos: BTreeMap<EntityId, usize> = BTreeMap::new();
        let pumping_refs = PumpingRefs {
            col_pumping_start: 0,
            pumping_stations: &no_stations,
            pumping_pos: &empty_pumping_pos,
        };
        resolve_variable_ref(
            &var_ref,
            block_idx,
            0, // stage_idx = 0
            indexer,
            production_models,
            &positions,
            &cascade_refs,
            &pumping_refs,
        )
    }

    /// Resolve a `PumpingFlow`/`PumpingPower` ref with an explicit pumping
    /// context (column start, block stride, station slice, position map).
    ///
    /// Threads real pumping data the way the matrix.rs caller does — sourcing
    /// `col_pumping_start` from a `StageLayout`-style reserved range — so the
    /// pumping arms exercise their real column arithmetic and consumption-rate
    /// coefficient instead of the empty fixture used by [`call`].
    #[allow(clippy::too_many_arguments)]
    fn call_pumping(
        var_ref: VariableRef,
        block_idx: usize,
        indexer: &StageIndexer,
        production_models: &ProductionModelSet,
        col_pumping_start: usize,
        n_blks: usize,
        pumping_stations: &[PumpingStation],
        pumping_pos: &BTreeMap<EntityId, usize>,
    ) -> Vec<(usize, f64)> {
        let empty: BTreeMap<EntityId, usize> = BTreeMap::new();
        let positions = super::EntityPositionMaps {
            hydro: &empty,
            thermal: &empty,
            bus: &empty,
            line: &empty,
        };
        let cascade = empty_cascade();
        let diversion_upstream: HashMap<EntityId, Vec<usize>> = HashMap::new();
        let cascade_refs = CascadeRefs {
            cascade: &cascade,
            diversion_upstream: &diversion_upstream,
        };
        let pumping_refs = PumpingRefs {
            col_pumping_start,
            pumping_stations,
            pumping_pos,
        };
        // The pumping column stride is now sourced from the indexer's `BlockGrid`,
        // so the fixture's declared `n_blks` must match `indexer.n_blks` for the
        // asserted columns to hold; pin that invariant rather than silently
        // diverging if a future fixture sets a mismatched stride.
        assert_eq!(n_blks, indexer.n_blks);
        resolve_variable_ref(
            &var_ref,
            block_idx,
            0, // stage_idx = 0
            indexer,
            production_models,
            &positions,
            &cascade_refs,
            &pumping_refs,
        )
    }

    /// A pumping station carrying a `consumption_mw_per_m3s` rate; every other
    /// field is an inert value the resolver does not read.
    fn make_pumping_station(id: i32, consumption_mw_per_m3s: f64) -> PumpingStation {
        PumpingStation {
            id: EntityId(id),
            name: String::new(),
            bus_id: EntityId(0),
            source_hydro_id: EntityId(0),
            destination_hydro_id: EntityId(0),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s,
            min_flow_m3s: 0.0,
            max_flow_m3s: 1.0,
        }
    }

    // ── ThermalGeneration tests ───────────────────────────────────────────────

    /// `ThermalGeneration` column arithmetic across the `block_id`/position axes
    /// the per-arm coverage requires: one `block_id = None`, one `block_id = Some`,
    /// and one `position != 0`. All resolve through `resolve_block_variable` with
    /// `block_col_range(indexer, ElementKind::Thermal).start = 49`, `n_blks = 3`.
    #[test]
    fn thermal_generation_column_arithmetic() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        // (case_name, thermal_id, block_id, block_idx, expected_col)
        let cases: [(&str, EntityId, Option<usize>, usize, usize); 3] = [
            ("none_block_1", EntityId(5), None, 1, 49 + 0 * 3 + 1),
            ("some_block_2", EntityId(5), Some(2), 2, 49 + 0 * 3 + 2),
            ("second_thermal", EntityId(6), None, 0, 49 + 1 * 3 + 0),
        ];

        for (case_name, thermal_id, block_id, block_idx, expected_col) in cases {
            let result = call(
                VariableRef::ThermalGeneration {
                    thermal_id,
                    block_id,
                },
                block_idx,
                &indexer,
                &prod,
                &hpos,
                &tpos,
                &bpos,
                &lpos,
            );
            assert_eq!(
                result,
                vec![(expected_col, 1.0)],
                "thermal_generation case `{case_name}`",
            );
        }
    }

    // ── HydroStorage tests ────────────────────────────────────────────────────

    /// HydroStorage returns stage-level storage column.
    ///
    /// storage.start = 0, hydro_pos[EntityId(10)] = 0
    /// Expected column = 0 + 0 = 0, regardless of block_idx.
    #[test]
    fn hydro_storage_stage_level_ignores_block() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        for block_idx in [0, 1, 2] {
            let result = call(
                VariableRef::HydroStorage {
                    hydro_id: EntityId(10),
                },
                block_idx,
                &indexer,
                &prod,
                &hpos,
                &tpos,
                &bpos,
                &lpos,
            );
            // storage.start = 0, pos = 0 → column 0
            assert_eq!(result, vec![(0, 1.0)], "block_idx={block_idx}");
        }

        // Hydro at position 2 (EntityId 30)
        let result2 = call(
            VariableRef::HydroStorage {
                hydro_id: EntityId(30),
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );
        // storage.start = 0, pos = 2 → column 2
        assert_eq!(result2, vec![(2, 1.0)]);
    }

    // ── HydroOutflow tests ────────────────────────────────────────────────────

    /// HydroOutflow returns 2 entries (turbine + spillage).
    ///
    /// hydro_pos[EntityId(40)] = 3 (position 3), block_id=None, block_idx=0
    /// turbine.start = 13, spillage.start = 25, n_blks = 3
    /// Expected: [(13 + 3*3 + 0, 1.0), (25 + 3*3 + 0, 1.0)] = [(22, 1.0), (34, 1.0)]
    #[test]
    fn hydro_outflow_expands_to_turbine_and_spillage() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::HydroOutflow {
                hydro_id: EntityId(40),
                block_id: None,
            },
            0, // block_idx
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        let turbine_col = 13 + 3 * 3 + 0; // 22
        let spillage_col = 25 + 3 * 3 + 0; // 34
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (turbine_col, 1.0));
        assert_eq!(result[1], (spillage_col, 1.0));
    }

    /// HydroOutflow with block_id=Some(1) at block_idx=0: should use the explicit block.
    #[test]
    fn hydro_outflow_block_id_some_uses_explicit_block() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::HydroOutflow {
                hydro_id: EntityId(10),
                block_id: Some(1),
            },
            0, // block_idx is irrelevant when block_id = Some
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        // hydro pos=0, turbine.start=13, spillage.start=25, block=1, n_blks=3
        assert_eq!(result, vec![(13 + 0 * 3 + 1, 1.0), (25 + 0 * 3 + 1, 1.0)]);
    }

    // ── HydroGeneration tests ─────────────────────────────────────────────────

    /// HydroGeneration for constant-productivity hydro returns
    /// turbine column with productivity multiplier.
    ///
    /// hydro_pos[EntityId(20)] = 1 → constant productivity 2.5
    /// turbine.start = 13, n_blks = 3, block_idx = 0
    /// Expected: [(13 + 1*3 + 0, 2.5)] = [(16, 2.5)]
    #[test]
    fn hydro_generation_constant_productivity_maps_to_turbine() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::HydroGeneration {
                hydro_id: EntityId(20),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        // hydro pos=1, turbine.start=13, n_blks=3, block=0, productivity=2.5
        assert_eq!(result, vec![(13 + 1 * 3 + 0, 2.5)]);
    }

    /// HydroGeneration for FPHA hydro returns generation column.
    ///
    /// hydro_pos[EntityId(10)] = 0 → FPHA (local FPHA index = 0)
    /// generation.start = 79, n_blks = 3, block_idx = 0
    /// Expected: [(79 + 0*3 + 0, 1.0)] = [(79, 1.0)]
    #[test]
    fn hydro_generation_fpha_maps_to_generation_column() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::HydroGeneration {
                hydro_id: EntityId(10),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        // FPHA local index 0, generation.start=79, n_blks=3, block=0
        assert_eq!(result, vec![(79 + 0 * 3 + 0, 1.0)]);
    }

    /// HydroGeneration for FPHA hydro at position 2 (second FPHA hydro, local index 1).
    ///
    /// hydro_pos[EntityId(30)] = 2 → FPHA (local FPHA index = 1)
    /// generation.start = 79, n_blks = 3, block_idx = 2
    /// Expected: [(79 + 1*3 + 2, 1.0)] = [(84, 1.0)]
    #[test]
    fn hydro_generation_fpha_second_hydro_block_2() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::HydroGeneration {
                hydro_id: EntityId(30),
                block_id: None,
            },
            2,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        // FPHA local index 1, generation.start=79, n_blks=3, block=2
        assert_eq!(result, vec![(79 + 1 * 3 + 2, 1.0)]);
    }

    // ── HydroEvaporation tests ────────────────────────────────────────────────

    /// HydroEvaporation maps to the evaporation-outflow column for the matching evaporation hydro.
    ///
    /// Use a dedicated indexer with evaporation hydros to test this path.
    ///
    /// N=2, L=0, T=0, Ln=0, B=1, K=1, no penalty, no FPHA, evap hydro at pos 0.
    /// theta = 2*(3+0) = 6
    /// turbine:    [7, 9)
    /// spillage:   [9, 11)
    /// diversion: [11, 13)
    /// deficit:   [13, 14)
    /// excess:    [14, 15)
    /// evap cols: [15, 18)  → evaporation_flow=15, f_evap_plus=16, f_evap_minus=17
    #[test]
    fn hydro_evaporation_maps_to_evaporation_flow_col() {
        let evap_indexer = StageIndexer::with_equipment_and_evaporation(
            &crate::indexer::EquipmentCounts {
                hydro_count: 2,
                max_par_order: 0,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                n_pumping: 0,
            },
            &crate::indexer::FphaColumnLayout {
                hydro_indices: vec![],
                planes_per_hydro: vec![],
            },
            &crate::indexer::EvapConfig {
                hydro_indices: vec![0],
            },
        );

        let prod_models = ProductionModelSet::new(
            vec![
                vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }],
                vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }],
            ],
            2,
            1,
        );

        let hpos: BTreeMap<EntityId, usize> =
            [(EntityId(10), 0), (EntityId(20), 1)].into_iter().collect();
        let tpos: BTreeMap<EntityId, usize> = BTreeMap::new();
        let bpos: BTreeMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
        let lpos: BTreeMap<EntityId, usize> = BTreeMap::new();

        let positions = super::EntityPositionMaps {
            hydro: &hpos,
            thermal: &tpos,
            bus: &bpos,
            line: &lpos,
        };
        let cascade = empty_cascade();
        let diversion_upstream: HashMap<EntityId, Vec<usize>> = HashMap::new();
        let cascade_refs = CascadeRefs {
            cascade: &cascade,
            diversion_upstream: &diversion_upstream,
        };
        let no_stations: Vec<PumpingStation> = Vec::new();
        let empty_pumping_pos: BTreeMap<EntityId, usize> = BTreeMap::new();
        let pumping_refs = PumpingRefs {
            col_pumping_start: 0,
            pumping_stations: &no_stations,
            pumping_pos: &empty_pumping_pos,
        };
        let result = resolve_variable_ref(
            &VariableRef::HydroEvaporation {
                hydro_id: EntityId(10),
            },
            0,
            0, // stage_idx
            &evap_indexer,
            &prod_models,
            &positions,
            &cascade_refs,
            &pumping_refs,
        );

        assert_eq!(result, vec![(15, 1.0)]);
    }

    /// HydroEvaporation for hydro that has no evaporation model returns empty vec.
    #[test]
    fn hydro_evaporation_no_evap_model_returns_empty() {
        let evap_indexer = StageIndexer::with_equipment_and_evaporation(
            &crate::indexer::EquipmentCounts {
                hydro_count: 2,
                max_par_order: 0,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                n_pumping: 0,
            },
            &crate::indexer::FphaColumnLayout {
                hydro_indices: vec![],
                planes_per_hydro: vec![],
            },
            &crate::indexer::EvapConfig {
                hydro_indices: vec![0],
            },
        );

        let prod_models = ProductionModelSet::new(
            vec![
                vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }],
                vec![ResolvedProductionModel::ConstantProductivity { productivity: 1.0 }],
            ],
            2,
            1,
        );

        let hpos: BTreeMap<EntityId, usize> =
            [(EntityId(10), 0), (EntityId(20), 1)].into_iter().collect();
        let tpos: BTreeMap<EntityId, usize> = BTreeMap::new();
        let bpos: BTreeMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
        let lpos: BTreeMap<EntityId, usize> = BTreeMap::new();

        // Hydro 20 (pos=1) has no evaporation in evap_hydro_indices=[0]
        let positions = super::EntityPositionMaps {
            hydro: &hpos,
            thermal: &tpos,
            bus: &bpos,
            line: &lpos,
        };
        let cascade = empty_cascade();
        let diversion_upstream: HashMap<EntityId, Vec<usize>> = HashMap::new();
        let cascade_refs = CascadeRefs {
            cascade: &cascade,
            diversion_upstream: &diversion_upstream,
        };
        let no_stations: Vec<PumpingStation> = Vec::new();
        let empty_pumping_pos: BTreeMap<EntityId, usize> = BTreeMap::new();
        let pumping_refs = PumpingRefs {
            col_pumping_start: 0,
            pumping_stations: &no_stations,
            pumping_pos: &empty_pumping_pos,
        };
        let result = resolve_variable_ref(
            &VariableRef::HydroEvaporation {
                hydro_id: EntityId(20),
            },
            0,
            0,
            &evap_indexer,
            &prod_models,
            &positions,
            &cascade_refs,
            &pumping_refs,
        );

        assert!(result.is_empty());
    }

    // ── Pumping tests ─────────────────────────────────────────────────────────
    //
    // Shared layout: two stations id 10 (p_idx 0, consumption 2.5 MW/(m³/s)) and
    // id 20 (p_idx 1, consumption 0.75), n_blks = 3, col_pumping_start = 100.
    // Block-major column = col_pumping_start + p_idx * n_blks + blk.

    const PUMP_COL_START: usize = 100;
    const PUMP_N_BLKS: usize = 3;

    /// Two pumping stations and the matching `pumping_pos`, in ID-sorted slot order.
    fn make_pumping_fixture() -> (Vec<PumpingStation>, BTreeMap<EntityId, usize>) {
        let stations = vec![
            make_pumping_station(10, 2.5),
            make_pumping_station(20, 0.75),
        ];
        let pumping_pos: BTreeMap<EntityId, usize> =
            [(EntityId(10), 0), (EntityId(20), 1)].into_iter().collect();
        (stations, pumping_pos)
    }

    /// `PumpingFlow{station, Some(blk)}` → the block-major flow column × 1.0.
    ///
    /// Station id 20 at p_idx 1, blk 2: col = 100 + 1*3 + 2 = 105, coeff 1.0.
    #[test]
    fn pumping_flow_resolves_to_flow_column_with_unit_coeff() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let (stations, ppos) = make_pumping_fixture();

        let result = call_pumping(
            VariableRef::PumpingFlow {
                station_id: EntityId(20),
                block_id: Some(2),
            },
            0, // block_idx — overridden by block_id = Some(2)
            &indexer,
            &prod,
            PUMP_COL_START,
            PUMP_N_BLKS,
            &stations,
            &ppos,
        );

        assert_eq!(result, vec![(PUMP_COL_START + 1 * PUMP_N_BLKS + 2, 1.0)]);
    }

    /// `PumpingPower{station, Some(blk)}` → the SAME flow column × consumption.
    ///
    /// Station id 10 at p_idx 0, blk 1: col = 100 + 0*3 + 1 = 101, coeff = 2.5.
    /// The column is identical to `PumpingFlow` for the same (station, blk) — the
    /// power term aliases the flow column, it is not a separate column.
    #[test]
    fn pumping_power_resolves_to_flow_column_with_consumption_coeff() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let (stations, ppos) = make_pumping_fixture();

        let blk = 1;
        let power = call_pumping(
            VariableRef::PumpingPower {
                station_id: EntityId(10),
                block_id: Some(blk),
            },
            0,
            &indexer,
            &prod,
            PUMP_COL_START,
            PUMP_N_BLKS,
            &stations,
            &ppos,
        );
        let flow = call_pumping(
            VariableRef::PumpingFlow {
                station_id: EntityId(10),
                block_id: Some(blk),
            },
            0,
            &indexer,
            &prod,
            PUMP_COL_START,
            PUMP_N_BLKS,
            &stations,
            &ppos,
        );

        let expected_col = PUMP_COL_START + 0 * PUMP_N_BLKS + blk;
        assert_eq!(power, vec![(expected_col, 2.5)]);
        // Same column as flow — PumpingPower must alias, not allocate a new column.
        assert_eq!(power[0].0, flow[0].0);
    }

    /// `PumpingFlow{station, None}` with `block_idx = k` resolves the single
    /// column for block `k` (`eff_blk = block_id.unwrap_or(block_idx)`), so the
    /// caller's per-block loop yields one `(col, 1.0)` entry per block in order.
    #[test]
    fn pumping_flow_none_resolves_per_block() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let (stations, ppos) = make_pumping_fixture();

        // The caller iterates blocks and calls the resolver once per block; collect
        // those per-block resolutions to confirm one (col, 1.0) entry per block.
        let per_block: Vec<(usize, f64)> = (0..PUMP_N_BLKS)
            .map(|blk| {
                let r = call_pumping(
                    VariableRef::PumpingFlow {
                        station_id: EntityId(10),
                        block_id: None,
                    },
                    blk, // block_idx supplies the effective block
                    &indexer,
                    &prod,
                    PUMP_COL_START,
                    PUMP_N_BLKS,
                    &stations,
                    &ppos,
                );
                assert_eq!(r.len(), 1);
                r[0]
            })
            .collect();

        assert_eq!(
            per_block,
            vec![
                (PUMP_COL_START + 0, 1.0),
                (PUMP_COL_START + 1, 1.0),
                (PUMP_COL_START + 2, 1.0),
            ]
        );
    }

    /// `PumpingPower{station, None}` resolves to the per-block column × consumption.
    ///
    /// Station id 20 at p_idx 1, consumption 0.75: per-block cols 103, 104, 105.
    #[test]
    fn pumping_power_none_resolves_per_block_with_consumption() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let (stations, ppos) = make_pumping_fixture();

        let per_block: Vec<(usize, f64)> = (0..PUMP_N_BLKS)
            .map(|blk| {
                let r = call_pumping(
                    VariableRef::PumpingPower {
                        station_id: EntityId(20),
                        block_id: None,
                    },
                    blk,
                    &indexer,
                    &prod,
                    PUMP_COL_START,
                    PUMP_N_BLKS,
                    &stations,
                    &ppos,
                );
                assert_eq!(r.len(), 1);
                r[0]
            })
            .collect();

        assert_eq!(
            per_block,
            vec![
                (PUMP_COL_START + 1 * PUMP_N_BLKS + 0, 0.75),
                (PUMP_COL_START + 1 * PUMP_N_BLKS + 1, 0.75),
                (PUMP_COL_START + 1 * PUMP_N_BLKS + 2, 0.75),
            ]
        );
    }

    /// Unknown station id resolves to `vec![]` for both pumping variants.
    #[test]
    fn pumping_unknown_station_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let (stations, ppos) = make_pumping_fixture();

        for var_ref in [
            VariableRef::PumpingFlow {
                station_id: EntityId(999),
                block_id: None,
            },
            VariableRef::PumpingPower {
                station_id: EntityId(999),
                block_id: Some(0),
            },
        ] {
            let result = call_pumping(
                var_ref,
                0,
                &indexer,
                &prod,
                PUMP_COL_START,
                PUMP_N_BLKS,
                &stations,
                &ppos,
            );
            assert!(
                result.is_empty(),
                "unknown station must return empty vec, got: {result:?} for {var_ref:?}"
            );
        }
    }

    /// `n_pumping == 0` (no stations) resolves to `vec![]` — the empty `pumping_pos`
    /// lookup misses before `col_pumping_start` is ever used.
    #[test]
    fn pumping_no_stations_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let no_stations: Vec<PumpingStation> = Vec::new();
        let empty_pos: BTreeMap<EntityId, usize> = BTreeMap::new();

        for var_ref in [
            VariableRef::PumpingFlow {
                station_id: EntityId(10),
                block_id: Some(0),
            },
            VariableRef::PumpingPower {
                station_id: EntityId(10),
                block_id: None,
            },
        ] {
            let result = call_pumping(
                var_ref,
                0,
                &indexer,
                &prod,
                PUMP_COL_START,
                PUMP_N_BLKS,
                &no_stations,
                &empty_pos,
            );
            assert!(
                result.is_empty(),
                "n_pumping == 0 must return empty vec, got: {result:?} for {var_ref:?}"
            );
        }
    }

    /// `PumpingFlow` and `PumpingPower` are block-DEPENDENT — per-block columns,
    /// so the single-row collapse must NOT apply (they stay in the `false` arm).
    #[test]
    fn pumping_variants_are_block_dependent() {
        assert!(!variable_ref_is_block_independent(
            &VariableRef::PumpingFlow {
                station_id: EntityId(10),
                block_id: None,
            }
        ));
        assert!(!variable_ref_is_block_independent(
            &VariableRef::PumpingPower {
                station_id: EntityId(10),
                block_id: None,
            }
        ));
    }

    /// `HydroStorage`, `HydroEvaporation`, and `AnticipatedDecision` are
    /// block-INDEPENDENT — stage-level stock variables whose resolver ignores
    /// `block_idx`, so the single-row collapse is sound. This is the `true`-arm
    /// counterpart to `pumping_variants_are_block_dependent` /
    /// `hydro_inflow_is_block_dependent`: dropping any of these three from the
    /// `true` branch of `variable_ref_is_block_independent` would silently expand
    /// a per-stage stock variable into per-block rows.
    #[test]
    fn block_independent_kinds_classify_true() {
        assert!(variable_ref_is_block_independent(
            &VariableRef::HydroStorage {
                hydro_id: EntityId(10),
            }
        ));
        assert!(variable_ref_is_block_independent(
            &VariableRef::HydroEvaporation {
                hydro_id: EntityId(10),
            }
        ));
        assert!(variable_ref_is_block_independent(
            &VariableRef::AnticipatedDecision {
                thermal_id: EntityId(6),
            }
        ));
    }

    // ── Stub entity tests ─────────────────────────────────────────────────────

    /// ContractImport returns empty vec.
    #[test]
    fn contract_import_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::ContractImport {
                contract_id: EntityId(99),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert!(result.is_empty());
    }

    /// ContractExport returns empty vec (the split-out export arm resolves through
    /// `block_col_range` to the empty `0..0` range, so no columns).
    #[test]
    fn contract_export_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::ContractExport {
                contract_id: EntityId(99),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert!(result.is_empty());
    }

    /// NonControllableGeneration returns empty vec.
    #[test]
    fn non_controllable_generation_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::NonControllableGeneration {
                source_id: EntityId(7),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert!(result.is_empty());
    }

    /// `HydroWithdrawal` resolves to an empty vec: withdrawal carries no LP
    /// decision column (a schedule fixed by bounds, not a decision variable), so
    /// a generic-constraint term referencing it contributes no `(column,
    /// coefficient)` pair — the deliberate stub contract documented above the
    /// `resolve_variable_ref` stub arm. `EntityId(999)` is in no `hydro_pos`
    /// entry, confirming the empty return is unconditional, not a missing-id
    /// fall-through.
    #[test]
    fn hydro_withdrawal_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::HydroWithdrawal {
                hydro_id: EntityId(999),
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(
            result,
            Vec::<(usize, f64)>::new(),
            "HydroWithdrawal must resolve to no column (no-LP-column stub contract)"
        );
    }

    /// `NonControllableCurtailment` resolves to an empty vec: non-controllable
    /// sources carry no decision column, so a generic-constraint term referencing
    /// curtailment contributes no `(column, coefficient)` pair — the same
    /// deliberate stub contract as `NonControllableGeneration`. `EntityId(999)` is
    /// in no position map, confirming the empty return is unconditional, not a
    /// missing-id fall-through.
    #[test]
    fn non_controllable_curtailment_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::NonControllableCurtailment {
                source_id: EntityId(999),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(
            result,
            Vec::<(usize, f64)>::new(),
            "NonControllableCurtailment must resolve to no column (no-LP-column stub contract)"
        );
    }

    // ── Missing entity ID test ─────────────────────────────────────────────────

    /// missing entity ID returns empty vec (defense-in-depth).
    #[test]
    fn missing_entity_id_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        // EntityId(999) is not in thermal_pos
        let result = call(
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(999),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert!(result.is_empty());
    }

    // ── BusDeficit tests ──────────────────────────────────────────────────────

    /// BusDeficit with S=2 deficit segments returns 2 column entries.
    ///
    /// bus_pos[EntityId(100)] = 0, deficit.start = 61, max_deficit_segments = 2,
    /// n_blks = 3, block_idx = 0
    /// Expected: [(61 + 0*2*3 + 0*3 + 0, 1.0), (61 + 0*2*3 + 1*3 + 0, 1.0)]
    ///         = [(61, 1.0), (64, 1.0)]
    #[test]
    fn bus_deficit_returns_one_entry_per_segment() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::BusDeficit {
                bus_id: EntityId(100),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        // deficit.start=61, b_pos=0, S=2, n_blks=3, blk=0
        // seg0: 61 + 0*2*3 + 0*3 + 0 = 61
        // seg1: 61 + 0*2*3 + 1*3 + 0 = 64
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (61, 1.0));
        assert_eq!(result[1], (64, 1.0));
    }

    /// BusDeficit for second bus (position 1) at block 1.
    ///
    /// bus_pos[EntityId(200)] = 1, deficit.start = 61, S = 2, n_blks = 3, blk = 1
    /// seg0: 61 + 1*2*3 + 0*3 + 1 = 61 + 6 + 0 + 1 = 68
    /// seg1: 61 + 1*2*3 + 1*3 + 1 = 61 + 6 + 3 + 1 = 71
    #[test]
    fn bus_deficit_second_bus_block_1() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::BusDeficit {
                bus_id: EntityId(200),
                block_id: None,
            },
            1,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result.len(), 2);
        assert_eq!(result[0], (68, 1.0));
        assert_eq!(result[1], (71, 1.0));
    }

    // ── BusExcess tests ───────────────────────────────────────────────────────

    /// BusExcess maps to the excess column for the bus.
    ///
    /// bus_pos[EntityId(100)] = 0, excess.start = 73, n_blks = 3, block = 2
    /// Expected: [(73 + 0*3 + 2, 1.0)] = [(75, 1.0)]
    #[test]
    fn bus_excess_maps_to_excess_column() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::BusExcess {
                bus_id: EntityId(100),
                block_id: None,
            },
            2,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(73 + 0 * 3 + 2, 1.0)]);
    }

    // ── LineDirect / LineReverse tests ────────────────────────────────────────

    /// LineDirect maps to line_fwd column.
    ///
    /// line_pos[EntityId(50)] = 0, line_fwd.start = 55, n_blks = 3, block = 1
    /// Expected: [(55 + 0*3 + 1, 1.0)] = [(56, 1.0)]
    #[test]
    fn line_direct_maps_to_fwd_column() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::LineDirect {
                line_id: EntityId(50),
                block_id: None,
            },
            1,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(55 + 0 * 3 + 1, 1.0)]);
    }

    /// LineReverse maps to line_rev column.
    ///
    /// line_pos[EntityId(50)] = 0, line_rev.start = 58, n_blks = 3, block = 0
    /// Expected: [(58 + 0*3 + 0, 1.0)] = [(58, 1.0)]
    #[test]
    fn line_reverse_maps_to_rev_column() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::LineReverse {
                line_id: EntityId(50),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(58, 1.0)]);
    }

    // ── LineExchange tests ──────────────────────────────────────────────────────

    /// LineExchange maps to both line_fwd and line_rev columns with opposite signs.
    ///
    /// line_pos[EntityId(50)] = 0, line_fwd.start = 55, line_rev.start = 58,
    /// n_blks = 3, block = 1
    /// Expected: [(55 + 0*3 + 1, 1.0), (58 + 0*3 + 1, -1.0)] = [(56, 1.0), (59, -1.0)]
    #[test]
    fn line_exchange_maps_to_fwd_and_rev_columns() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::LineExchange {
                line_id: EntityId(50),
                block_id: None,
            },
            1,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(56, 1.0), (59, -1.0)]);
    }

    /// LineExchange with explicit block_id overrides current block_idx.
    ///
    /// block_idx = 2 but block_id = Some(0)
    /// Expected: [(55, 1.0), (58, -1.0)]
    #[test]
    fn line_exchange_with_explicit_block() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::LineExchange {
                line_id: EntityId(50),
                block_id: Some(0),
            },
            2,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(55, 1.0), (58, -1.0)]);
    }

    /// LineExchange with unknown line ID returns empty vec.
    #[test]
    fn line_exchange_unknown_id_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::LineExchange {
                line_id: EntityId(999),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert!(result.is_empty());
    }

    // ── AnticipatedDecision tests ─────────────────────────────────────────────

    /// Build a `StageIndexer` with 2 thermals where thermal at system position 1
    /// is anticipated (local anticipated index 0).
    ///
    /// N=0 hydros, T=2 thermals (pos 0 = regular, pos 1 = anticipated), Ln=0,
    /// B=1 bus, K=2 blocks, n_anticipated=1, k_max=2.
    ///
    /// Column layout (no hydros, no FPHA, no evap):
    ///   storage:            [0, 0)    empty
    ///   lags:               [0, 0)    empty
    ///   z_inflow:           [0, 0)    empty
    ///   storage_in:         [0, 0)    empty
    ///   theta:              0
    ///   decision_start:     1
    ///   anticipated_state:  [1, 1 + 2*1) = [1, 3)  (k_max=2, n_anticipated=1)
    ///   thermal:            [3, 3 + 2*2) = [3, 7)  (T=2, K=2)
    ///   anticipated_decision: [7, 7+1) = [7, 8)   (n_anticipated=1)
    ///   line_fwd: [8, 8) empty
    ///   line_rev: [8, 8) empty
    ///   deficit: [8, 8+1*1*2) = [8, 10)  (B=1, S=1, K=2)
    ///   excess:  [10, 10+1*2) = [10, 12)
    fn make_indexer_with_anticipated() -> StageIndexer {
        StageIndexer::with_equipment_and_evaporation(
            &crate::indexer::EquipmentCounts {
                hydro_count: 0,
                max_par_order: 0,
                n_thermals: 2,
                n_lines: 0,
                n_buses: 1,
                n_blks: 2,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 1,
                k_max: 2,
                anticipated_lead_stages: vec![2],
                anticipated_thermal_indices: vec![1], // sys pos 1 is anticipated
                n_pumping: 0,
            },
            &crate::indexer::FphaColumnLayout {
                hydro_indices: vec![],
                planes_per_hydro: vec![],
            },
            &crate::indexer::EvapConfig {
                hydro_indices: vec![],
            },
        )
    }

    /// AC-12: `AnticipatedDecision` for an anticipated thermal maps to the
    /// correct stage-level column: `anticipated_decision.start + local_idx`.
    ///
    /// Using `make_indexer_with_anticipated`:
    /// - Thermal EntityId(6) at sys_pos=1, which is anticipated_thermal_indices[0].
    /// - anticipated_decision.start = 7, local_idx = 0.
    /// - Expected column = 7 + 0 = 7.
    #[test]
    fn anticipated_decision_maps_to_correct_column() {
        let indexer = make_indexer_with_anticipated();
        let prod = ProductionModelSet::new(vec![], 0, 1);
        let hpos: BTreeMap<EntityId, usize> = BTreeMap::new();
        let tpos: BTreeMap<EntityId, usize> =
            [(EntityId(5), 0), (EntityId(6), 1)].into_iter().collect();
        let bpos: BTreeMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
        let lpos: BTreeMap<EntityId, usize> = BTreeMap::new();

        // Verify anticipated_decision.start is as expected.
        assert_eq!(
            indexer.anticipated_decision.start, 7,
            "anticipated_decision.start should be 7, got {}",
            indexer.anticipated_decision.start
        );

        let result = call(
            VariableRef::AnticipatedDecision {
                thermal_id: EntityId(6), // sys_pos=1, local anticipated idx=0
            },
            0, // block_idx is ignored for stage-level variable
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(
            result,
            vec![(7, 1.0)],
            "AnticipatedDecision(6) should resolve to column 7 (anticipated_decision.start + 0)"
        );
    }

    /// AC-12 (block-independence): `AnticipatedDecision` is stage-level — the
    /// returned column is the same regardless of `block_idx`.
    #[test]
    fn anticipated_decision_ignores_block_idx() {
        let indexer = make_indexer_with_anticipated();
        let prod = ProductionModelSet::new(vec![], 0, 1);
        let hpos: BTreeMap<EntityId, usize> = BTreeMap::new();
        let tpos: BTreeMap<EntityId, usize> =
            [(EntityId(5), 0), (EntityId(6), 1)].into_iter().collect();
        let bpos: BTreeMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
        let lpos: BTreeMap<EntityId, usize> = BTreeMap::new();

        for block_idx in [0, 1] {
            let result = call(
                VariableRef::AnticipatedDecision {
                    thermal_id: EntityId(6),
                },
                block_idx,
                &indexer,
                &prod,
                &hpos,
                &tpos,
                &bpos,
                &lpos,
            );
            assert_eq!(
                result,
                vec![(7, 1.0)],
                "AnticipatedDecision must be stage-level (block_idx={block_idx} should not change column)"
            );
        }
    }

    /// AC-13: `AnticipatedDecision` for a regular (non-anticipated) thermal
    /// returns an empty vec (defense-in-depth).
    ///
    /// Thermal EntityId(5) at sys_pos=0 is NOT in anticipated_thermal_indices.
    #[test]
    fn anticipated_decision_non_anticipated_thermal_returns_empty() {
        let indexer = make_indexer_with_anticipated();
        let prod = ProductionModelSet::new(vec![], 0, 1);
        let hpos: BTreeMap<EntityId, usize> = BTreeMap::new();
        let tpos: BTreeMap<EntityId, usize> =
            [(EntityId(5), 0), (EntityId(6), 1)].into_iter().collect();
        let bpos: BTreeMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
        let lpos: BTreeMap<EntityId, usize> = BTreeMap::new();

        let result = call(
            VariableRef::AnticipatedDecision {
                thermal_id: EntityId(5), // sys_pos=0, NOT anticipated
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert!(
            result.is_empty(),
            "AnticipatedDecision for non-anticipated thermal must return empty vec, got: {result:?}"
        );
    }

    /// AC-14: `AnticipatedDecision` for an unknown entity ID returns empty vec.
    #[test]
    fn anticipated_decision_unknown_entity_returns_empty() {
        let indexer = make_indexer_with_anticipated();
        let prod = ProductionModelSet::new(vec![], 0, 1);
        let hpos: BTreeMap<EntityId, usize> = BTreeMap::new();
        let tpos: BTreeMap<EntityId, usize> =
            [(EntityId(5), 0), (EntityId(6), 1)].into_iter().collect();
        let bpos: BTreeMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
        let lpos: BTreeMap<EntityId, usize> = BTreeMap::new();

        let result = call(
            VariableRef::AnticipatedDecision {
                thermal_id: EntityId(999), // unknown
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert!(
            result.is_empty(),
            "AnticipatedDecision for unknown entity must return empty vec, got: {result:?}"
        );
    }

    // ── block_col_range tests ─────────────────────────────────────────────────

    /// Each equipment/line family maps to its matching `StageIndexer` range, and
    /// the two contract families map to the empty `0..0` sentinel. This pins the
    /// family↔range pairing the resolver's `col_start` reads depend on.
    #[test]
    fn block_col_range_maps_each_family_to_its_indexer_range() {
        let idx = make_indexer();

        assert_eq!(block_col_range(&idx, ElementKind::Turbine), idx.turbine);
        assert_eq!(block_col_range(&idx, ElementKind::Spillage), idx.spillage);
        assert_eq!(block_col_range(&idx, ElementKind::Diversion), idx.diversion);
        assert_eq!(block_col_range(&idx, ElementKind::Thermal), idx.thermal);
        assert_eq!(block_col_range(&idx, ElementKind::LineFwd), idx.line_fwd);
        assert_eq!(block_col_range(&idx, ElementKind::LineRev), idx.line_rev);
        assert_eq!(block_col_range(&idx, ElementKind::Excess), idx.excess);

        assert_eq!(block_col_range(&idx, ElementKind::ContractImport), 0..0);
        assert_eq!(block_col_range(&idx, ElementKind::ContractExport), 0..0);
    }

    // ── HydroTurbined / HydroSpillage tests ───────────────────────────────────

    /// HydroTurbined maps to turbine column.
    #[test]
    fn hydro_turbined_maps_to_turbine_column() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        // hydro pos=1 (EntityId 20), turbine.start=13, n_blks=3, block=2
        let result = call(
            VariableRef::HydroTurbined {
                hydro_id: EntityId(20),
                block_id: None,
            },
            2,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(13 + 1 * 3 + 2, 1.0)]);
    }

    /// HydroSpillage maps to spillage column.
    #[test]
    fn hydro_spillage_maps_to_spillage_column() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        // hydro pos=3 (EntityId 40), spillage.start=25, n_blks=3, block=1
        let result = call(
            VariableRef::HydroSpillage {
                hydro_id: EntityId(40),
                block_id: None,
            },
            1,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(25 + 3 * 3 + 1, 1.0)]);
    }

    /// HydroDiversion maps to the diversion column.
    ///
    /// Routes through `resolve_block_variable` with
    /// `block_col_range(indexer, ElementKind::Diversion).start = 37`. For hydro
    /// pos=1 (EntityId 20), n_blks=3, block=2 the flat block-major address is
    /// `37 + 1*3 + 2 = 42` with the unit coefficient.
    #[test]
    fn diversion_maps_to_diversion_column() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::HydroDiversion {
                hydro_id: EntityId(20),
                block_id: None,
            },
            2,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(37 + 1 * 3 + 2, 1.0)]);
    }

    // ── HydroInflow tests ──────────────────────────────────────────────────────

    /// Cascade for the total-inflow tests: EntityId(10) and EntityId(20) both
    /// flow into EntityId(40), so `upstream(40) = [10, 20]` (ID-sorted). The
    /// three hydros map to system positions 0, 1, 3 via `make_hydro_pos`.
    fn make_inflow_cascade() -> CascadeTopology {
        CascadeTopology::build(&[
            make_hydro(10, Some(40)),
            make_hydro(20, Some(40)),
            make_hydro(30, None),
            make_hydro(40, None),
        ])
    }

    /// AC: a two-upstream hydro (no diversion-into) resolves at block `k` to the
    /// incremental `z_inflow` column plus, in canonical upstream order, each
    /// upstream plant's turbine + spillage column. All coefficients `+1.0`.
    ///
    /// Target EntityId(40) at pos_h=3; upstream [10, 20] at pos 0, 1.
    /// z_inflow.start=4 → (4+3, 1.0)=(7, 1.0); turbine.start=13, spillage.start=25,
    /// n_blks=3, k=2.
    #[test]
    fn hydro_inflow_two_upstream_canonical_order() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();
        let cascade = make_inflow_cascade();
        let div: HashMap<EntityId, Vec<usize>> = HashMap::new();

        let blk = 2;
        let result = call_with_cascade(
            VariableRef::HydroInflow {
                hydro_id: EntityId(40),
                block_id: Some(blk),
            },
            0, // block_idx — overridden by block_id = Some(blk)
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
            &cascade,
            &div,
        );

        let z_col = 4 + 3; // z_inflow.start + pos_h
        let turb = 13; // turbine.start
        let spill = 25; // spillage.start
        let nb = 3; // n_blks
        assert_eq!(
            result,
            vec![
                (z_col, 1.0),
                (turb + 0 * nb + blk, 1.0),  // upstream 10 turbine
                (spill + 0 * nb + blk, 1.0), // upstream 10 spillage
                (turb + 1 * nb + blk, 1.0),  // upstream 20 turbine
                (spill + 1 * nb + blk, 1.0), // upstream 20 spillage
            ]
        );
    }

    /// AC: `block_id = None` with `block_idx = k` matches `block_id = Some(k)`
    /// (the resolver uses `eff_blk = block_id.unwrap_or(block_idx)`).
    #[test]
    fn hydro_inflow_none_matches_some_block_idx() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();
        let cascade = make_inflow_cascade();
        let div: HashMap<EntityId, Vec<usize>> = HashMap::new();

        let blk = 2;
        let none_result = call_with_cascade(
            VariableRef::HydroInflow {
                hydro_id: EntityId(40),
                block_id: None,
            },
            blk, // block_idx supplies the effective block
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
            &cascade,
            &div,
        );
        let some_result = call_with_cascade(
            VariableRef::HydroInflow {
                hydro_id: EntityId(40),
                block_id: Some(blk),
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
            &cascade,
            &div,
        );

        assert_eq!(none_result, some_result);
    }

    /// AC: a hydro that also has a plant diverting into it gets the diversion
    /// column appended after the upstream terms.
    ///
    /// `diversion_upstream[40] = [2]` (system index 2). diversion.start=37,
    /// n_blks=3, k=1 → (37 + 2*3 + 1, 1.0) = (44, 1.0).
    #[test]
    fn hydro_inflow_diversion_into_appends_diversion_column() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();
        let cascade = make_inflow_cascade();
        let div: HashMap<EntityId, Vec<usize>> = [(EntityId(40), vec![2])].into_iter().collect();

        let blk = 1;
        let result = call_with_cascade(
            VariableRef::HydroInflow {
                hydro_id: EntityId(40),
                block_id: Some(blk),
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
            &cascade,
            &div,
        );

        let z_col = 4 + 3;
        let turb = 13; // turbine.start
        let spill = 25; // spillage.start
        let div_start = 37; // diversion.start
        let nb = 3; // n_blks
        assert_eq!(
            result,
            vec![
                (z_col, 1.0),
                (turb + 0 * nb + blk, 1.0),
                (spill + 0 * nb + blk, 1.0),
                (turb + 1 * nb + blk, 1.0),
                (spill + 1 * nb + blk, 1.0),
                (div_start + 2 * nb + blk, 1.0), // diversion-into, system index 2
            ]
        );
    }

    /// AC: a headwater hydro (no upstream, no diversion-into) resolves to exactly
    /// the incremental `z_inflow` column.
    ///
    /// EntityId(30) at pos=2 is a headwater in `make_inflow_cascade`.
    /// z_inflow.start=4 → (6, 1.0).
    #[test]
    fn hydro_inflow_headwater_resolves_to_z_inflow_only() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();
        let cascade = make_inflow_cascade();
        let div: HashMap<EntityId, Vec<usize>> = HashMap::new();

        for block_idx in [0, 1, 2] {
            let result = call_with_cascade(
                VariableRef::HydroInflow {
                    hydro_id: EntityId(30),
                    block_id: None,
                },
                block_idx,
                &indexer,
                &prod,
                &hpos,
                &tpos,
                &bpos,
                &lpos,
                &cascade,
                &div,
            );
            assert_eq!(result, vec![(6, 1.0)], "block_idx={block_idx}");
        }
    }

    /// AC: `hydro_count == 0` (empty `z_inflow`) resolves to `vec![]`.
    ///
    /// `make_indexer_with_anticipated` has no hydros, so `z_inflow` is empty and
    /// `z_inflow.start` is meaningless; the resolver must short-circuit to `[]`.
    #[test]
    fn hydro_inflow_empty_when_no_hydros() {
        let indexer = make_indexer_with_anticipated();
        let prod = ProductionModelSet::new(vec![], 0, 1);
        let hpos: BTreeMap<EntityId, usize> = BTreeMap::new();
        let tpos: BTreeMap<EntityId, usize> = BTreeMap::new();
        let bpos: BTreeMap<EntityId, usize> = BTreeMap::new();
        let lpos: BTreeMap<EntityId, usize> = BTreeMap::new();

        assert!(
            indexer.z_inflow.is_empty(),
            "z_inflow must be empty with hydro_count == 0"
        );

        let result = call(
            VariableRef::HydroInflow {
                hydro_id: EntityId(0),
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert!(
            result.is_empty(),
            "HydroInflow with hydro_count == 0 must return empty vec, got: {result:?}"
        );
    }

    /// AC: an unknown `hydro_id` resolves to `vec![]`.
    #[test]
    fn hydro_inflow_unknown_id_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::HydroInflow {
                hydro_id: EntityId(999), // unknown
                block_id: None,
            },
            0,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert!(
            result.is_empty(),
            "HydroInflow for unknown id must return empty vec, got: {result:?}"
        );
    }

    /// AC: `HydroInflow` is block-DEPENDENT — its upstream releases are per-block
    /// columns, so the single-row collapse must NOT apply.
    #[test]
    fn hydro_inflow_is_block_dependent() {
        assert!(!variable_ref_is_block_independent(
            &VariableRef::HydroInflow {
                hydro_id: EntityId(0),
                block_id: None,
            }
        ));
    }
}
