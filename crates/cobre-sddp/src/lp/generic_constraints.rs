//! Variable reference to LP column index mapping for generic constraints.
//!
//! This module provides `resolve_variable_ref`, which maps a [`VariableRef`]
//! and block index to a list of `(column_index, coefficient_multiplier)` pairs.
//! The LP builder calls this function for each [`cobre_core::LinearTerm`] in a
//! generic constraint expression to produce the CSC matrix entries.
//!
//! ## Column index arithmetic
//!
//! All column offsets follow the layout defined in [`StageIndexer`]:
//!
//! - Block-level variables (turbine, spillage, thermal, line, deficit, excess)
//!   use `col_start + entity_pos * n_blks + block_idx`.
//! - Stage-level variables (storage, evaporation, withdrawal) use `col_start + entity_pos`.
//! - FPHA generation uses `generation.start + fpha_local_idx * n_blks + block_idx`.
//!
//! ## Block expansion
//!
//! When `block_id = None` for a block-level variable, the function returns the
//! column for the *current* `block_idx` rather than expanding to all blocks.
//! The caller iterates over blocks and calls this function once per block, so
//! the per-block expansion happens in the caller loop, not here.
//!
//! ## Stub entities
//!
//! Variables that reference entity types with no LP columns (pumping stations,
//! contracts, non-controllable sources) return an empty vec. This is consistent
//! with the convention that the constraint term has no LP effect for those
//! entity types.

use std::collections::HashMap;
use std::hash::BuildHasher;

use cobre_core::{CascadeTopology, ConstraintExpression, EntityId, VariableRef};

use crate::hydro_models::{ProductionModelSet, ResolvedProductionModel};
use crate::indexer::StageIndexer;

/// Position maps for entity types, mapping entity IDs to their index in
/// the system's entity arrays.
///
/// Used by [`resolve_variable_ref`] to translate `VariableRef` entity IDs
/// into LP column offsets.
pub(crate) struct EntityPositionMaps<'a, S: BuildHasher = std::hash::RandomState> {
    /// Hydro plant ID to position index.
    pub hydro: &'a HashMap<EntityId, usize, S>,
    /// Thermal unit ID to position index.
    pub thermal: &'a HashMap<EntityId, usize, S>,
    /// Bus ID to position index.
    pub bus: &'a HashMap<EntityId, usize, S>,
    /// Line ID to position index.
    pub line: &'a HashMap<EntityId, usize, S>,
}

/// Borrowed cascade context for resolving the total-inflow expression.
///
/// Grouped (rather than passed as two more positional arguments) so
/// [`resolve_variable_ref`] does not trip `clippy::too_many_arguments`,
/// mirroring the [`EntityPositionMaps`] grouping idiom. The `HydroInflow` arm is
/// the only consumer; every other arm ignores it.
pub(crate) struct CascadeRefs<'a, S: BuildHasher = std::hash::RandomState> {
    /// Immediately-upstream cascade adjacency. `upstream(h)` returns
    /// `&[EntityId]` sorted by `EntityId.0` at build time, so the resolver
    /// iterates it in a fixed, input-ordering-independent sequence.
    pub cascade: &'a CascadeTopology,
    /// Target hydro id to the **system indices** of plants diverting into it,
    /// built in canonical hydro order (the same representation `matrix.rs`
    /// iterates for the storage-balance diversion-inflow term).
    pub diversion_upstream: &'a HashMap<EntityId, Vec<usize>, S>,
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
/// - `n_blks` — number of operating blocks in this stage.
/// - `stage_idx` — stage index used to look up per-stage production models.
/// - `indexer` — column layout for the current stage LP.
/// - `production_models` — resolved production model set, used to distinguish
///   FPHA hydros from constant-productivity hydros for `HydroGeneration`.
/// - `positions` — entity position maps grouped into [`EntityPositionMaps`].
/// - `cascade_refs` — cascade topology + diversion-into map grouped into
///   [`CascadeRefs`]; consulted only by the `HydroInflow` arm.
///
/// # Returns
///
/// An empty vec when:
/// - The entity ID is not found in the relevant position map (should have been
///   caught by referential validation, but this is defense-in-depth).
/// - The variable type references a stub entity with no LP columns (pumping
///   stations, contracts, non-controllable sources, diversion, withdrawal).
#[must_use]
pub(crate) fn resolve_variable_ref<S: BuildHasher>(
    var_ref: &VariableRef,
    block_idx: usize,
    n_blks: usize,
    stage_idx: usize,
    indexer: &StageIndexer,
    production_models: &ProductionModelSet,
    positions: &EntityPositionMaps<'_, S>,
    cascade_refs: &CascadeRefs<'_, S>,
) -> Vec<(usize, f64)> {
    let hydro_pos = positions.hydro;
    let thermal_pos = positions.thermal;
    let bus_pos = positions.bus;
    let line_pos = positions.line;
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
            n_blks,
            indexer,
            hydro_pos,
            cascade_refs,
        ),

        VariableRef::HydroTurbined { hydro_id, block_id } => resolve_block_variable(
            *hydro_id,
            *block_id,
            block_idx,
            n_blks,
            indexer.turbine.start,
            hydro_pos,
            1.0,
        ),

        VariableRef::HydroSpillage { hydro_id, block_id } => resolve_block_variable(
            *hydro_id,
            *block_id,
            block_idx,
            n_blks,
            indexer.spillage.start,
            hydro_pos,
            1.0,
        ),

        VariableRef::HydroOutflow { hydro_id, block_id } => {
            resolve_hydro_outflow(*hydro_id, *block_id, block_idx, n_blks, indexer, hydro_pos)
        }

        VariableRef::HydroGeneration { hydro_id, block_id } => resolve_hydro_generation(
            *hydro_id,
            *block_id,
            block_idx,
            n_blks,
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
            n_blks,
            indexer.thermal.start,
            thermal_pos,
            1.0,
        ),

        // ── Transmission lines ─────────────────────────────────────────────
        VariableRef::LineDirect { line_id, block_id } => resolve_block_variable(
            *line_id,
            *block_id,
            block_idx,
            n_blks,
            indexer.line_fwd.start,
            line_pos,
            1.0,
        ),

        VariableRef::LineReverse { line_id, block_id } => resolve_block_variable(
            *line_id,
            *block_id,
            block_idx,
            n_blks,
            indexer.line_rev.start,
            line_pos,
            1.0,
        ),

        VariableRef::LineExchange { line_id, block_id } => {
            resolve_line_exchange(*line_id, *block_id, block_idx, n_blks, indexer, line_pos)
        }

        // ── Bus deficit / excess ───────────────────────────────────────────
        VariableRef::BusDeficit { bus_id, block_id } => {
            resolve_bus_deficit(*bus_id, *block_id, block_idx, n_blks, indexer, bus_pos)
        }

        VariableRef::BusExcess { bus_id, block_id } => resolve_block_variable(
            *bus_id,
            *block_id,
            block_idx,
            n_blks,
            indexer.excess.start,
            bus_pos,
            1.0,
        ),

        VariableRef::HydroDiversion { hydro_id, block_id } => resolve_block_variable(
            *hydro_id,
            *block_id,
            block_idx,
            n_blks,
            indexer.diversion.start,
            hydro_pos,
            1.0,
        ),

        // ── Anticipated thermal decision column ────────────────────────────
        VariableRef::AnticipatedDecision { thermal_id } => {
            resolve_anticipated_decision(*thermal_id, indexer, thermal_pos)
        }

        // ── Stub entities with no LP columns ──────────────────────────────
        // The following entity types are registered in the data model but do not
        // contribute LP decision variables in this implementation:
        // - HydroWithdrawal: withdrawal is a schedule fixed by bounds, not a
        //   decision variable.
        // - PumpingFlow, PumpingPower: pumping stations are NO-OP stubs.
        // - ContractImport, ContractExport: contracts are NO-OP stubs.
        // - NonControllableGeneration, NonControllableCurtailment: non-controllable
        //   sources are NO-OP stubs.
        VariableRef::HydroWithdrawal { .. }
        | VariableRef::PumpingFlow { .. }
        | VariableRef::PumpingPower { .. }
        | VariableRef::ContractImport { .. }
        | VariableRef::ContractExport { .. }
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
/// Every other kind is block-level: it resolves to `col_start + pos * n_blks +
/// block_idx` (via `resolve_block_variable` / `resolve_hydro_inflow` / FPHA
/// generation / line / deficit / excess), so distinct block indices yield
/// distinct columns. The stub kinds (withdrawal, pumping, contracts,
/// non-controllable) resolve to no columns at all; they are conservatively
/// treated as block-level here so that only *provably* stock variables enable the
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
fn resolve_hydro_storage<S: BuildHasher>(
    hydro_id: EntityId,
    indexer: &StageIndexer,
    hydro_pos: &HashMap<EntityId, usize, S>,
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
/// slacks, AR-lag-`ψ`, and pumped inflow (a NO-OP stub until pumping columns
/// exist) are **excluded**: they are storage/loss/outflow terms or have no LP
/// column.
///
/// Column arithmetic mirrors [`resolve_block_variable`]:
/// `col_start + pos * n_blks + eff_blk` with `eff_blk = block_id.unwrap_or(block_idx)`.
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
fn resolve_hydro_inflow<S: BuildHasher>(
    hydro_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    n_blks: usize,
    indexer: &StageIndexer,
    hydro_pos: &HashMap<EntityId, usize, S>,
    cascade_refs: &CascadeRefs<'_, S>,
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
    if !indexer.turbine.is_empty() && !indexer.spillage.is_empty() {
        for &up_id in upstream {
            if let Some(&pos_up) = hydro_pos.get(&up_id) {
                result.push((indexer.turbine.start + pos_up * n_blks + eff_blk, 1.0));
                result.push((indexer.spillage.start + pos_up * n_blks + eff_blk, 1.0));
            }
        }
    }

    // Diverted inflow: the diversion column of each plant diverting into `h`.
    // `diversion_upstream[h]` already holds system indices, so no `hydro_pos`
    // lookup is needed (mirrors the matrix.rs diversion-inflow loop).
    if !indexer.diversion.is_empty() {
        for &d_idx in diversion_into {
            result.push((indexer.diversion.start + d_idx * n_blks + eff_blk, 1.0));
        }
    }

    result
}

/// Resolve `HydroEvaporation` to the evaporation-outflow column for the matching hydro.
///
/// The evaporation list uses a local index; we find it by matching the
/// system-level hydro position. Returns empty vec when the hydro has no
/// linearized evaporation at this stage.
fn resolve_hydro_evaporation<S: BuildHasher>(
    hydro_id: EntityId,
    indexer: &StageIndexer,
    hydro_pos: &HashMap<EntityId, usize, S>,
) -> Vec<(usize, f64)> {
    if let Some(&sys_pos) = hydro_pos.get(&hydro_id) {
        if let Some(local_idx) = indexer
            .evap_hydro_indices
            .iter()
            .position(|&p| p == sys_pos)
        {
            let evaporation_flow_col = indexer.evap_indices[local_idx].evaporation_flow_col;
            vec![(evaporation_flow_col, 1.0)]
        } else {
            vec![]
        }
    } else {
        vec![]
    }
}

/// Resolve `HydroOutflow` (turbine + spillage) to two block-level columns.
fn resolve_hydro_outflow<S: BuildHasher>(
    hydro_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    n_blks: usize,
    indexer: &StageIndexer,
    hydro_pos: &HashMap<EntityId, usize, S>,
) -> Vec<(usize, f64)> {
    let mut result = Vec::with_capacity(2);
    result.extend(resolve_block_variable(
        hydro_id,
        block_id,
        block_idx,
        n_blks,
        indexer.turbine.start,
        hydro_pos,
        1.0,
    ));
    result.extend(resolve_block_variable(
        hydro_id,
        block_id,
        block_idx,
        n_blks,
        indexer.spillage.start,
        hydro_pos,
        1.0,
    ));
    result
}

/// Resolve `HydroGeneration` by dispatching on the production model.
///
/// - FPHA hydros: maps to the generation column at `fpha_local_idx * n_blks + blk`.
/// - Constant-productivity hydros: maps to the turbine column scaled by productivity.
fn resolve_hydro_generation<S: BuildHasher>(
    hydro_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    n_blks: usize,
    stage_idx: usize,
    indexer: &StageIndexer,
    production_models: &ProductionModelSet,
    hydro_pos: &HashMap<EntityId, usize, S>,
) -> Vec<(usize, f64)> {
    let Some(&sys_pos) = hydro_pos.get(&hydro_id) else {
        return vec![];
    };
    match production_models.model(sys_pos, stage_idx) {
        ResolvedProductionModel::Fpha { .. } => {
            if let Some(fpha_local_idx) = indexer
                .fpha_hydro_indices
                .iter()
                .position(|&p| p == sys_pos)
            {
                let effective_blk = block_id.unwrap_or(block_idx);
                let col = indexer.generation.start + fpha_local_idx * n_blks + effective_blk;
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
                n_blks,
                indexer.turbine.start,
                hydro_pos,
                *productivity,
            )
        }
    }
}

/// Resolve `LineExchange` (net = forward − reverse) to two columns with signs.
fn resolve_line_exchange<S: BuildHasher>(
    line_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    n_blks: usize,
    indexer: &StageIndexer,
    line_pos: &HashMap<EntityId, usize, S>,
) -> Vec<(usize, f64)> {
    if let Some(&pos) = line_pos.get(&line_id) {
        let effective_blk = block_id.unwrap_or(block_idx);
        let fwd_col = indexer.line_fwd.start + pos * n_blks + effective_blk;
        let rev_col = indexer.line_rev.start + pos * n_blks + effective_blk;
        vec![(fwd_col, 1.0), (rev_col, -1.0)]
    } else {
        vec![]
    }
}

/// Resolve `BusDeficit` to one column per deficit segment.
///
/// Column layout: `deficit.start + b_pos * S * n_blks + seg * n_blks + blk`.
fn resolve_bus_deficit<S: BuildHasher>(
    bus_id: EntityId,
    block_id: Option<usize>,
    block_idx: usize,
    n_blks: usize,
    indexer: &StageIndexer,
    bus_pos: &HashMap<EntityId, usize, S>,
) -> Vec<(usize, f64)> {
    if let Some(&b_pos) = bus_pos.get(&bus_id) {
        let effective_blk = block_id.unwrap_or(block_idx);
        let s = indexer.max_deficit_segments;
        let base = indexer.deficit.start + b_pos * s * n_blks + effective_blk;
        (0..s).map(|seg| (base + seg * n_blks, 1.0)).collect()
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
fn resolve_anticipated_decision<S: BuildHasher>(
    thermal_id: EntityId,
    indexer: &StageIndexer,
    thermal_pos: &HashMap<EntityId, usize, S>,
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

/// Resolve a block-level LP variable to a `(column_index, multiplier)` pair.
///
/// Computes `col_start + entity_pos * n_blks + effective_block_idx` where
/// `effective_block_idx` is `block_idx` when `ref_block_id` is `None`, or
/// `b` when `ref_block_id` is `Some(b)`.
///
/// Returns an empty vec if the entity ID is not found in `pos_map`.
fn resolve_block_variable<S: BuildHasher>(
    entity_id: EntityId,
    ref_block_id: Option<usize>,
    current_block_idx: usize,
    n_blks: usize,
    col_start: usize,
    pos_map: &HashMap<EntityId, usize, S>,
    multiplier: f64,
) -> Vec<(usize, f64)> {
    if let Some(&pos) = pos_map.get(&entity_id) {
        let effective_blk = ref_block_id.unwrap_or(current_block_idx);
        vec![(col_start + pos * n_blks + effective_blk, multiplier)]
    } else {
        vec![]
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
    use std::collections::HashMap;

    use cobre_core::entities::{HydroGenerationModel, HydroPenalties};
    use cobre_core::{CascadeTopology, EntityId, Hydro, VariableRef};

    use super::{CascadeRefs, resolve_variable_ref, variable_ref_is_block_independent};
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
    /// Column layout (NEW: z_inflow at [N*(1+L), N*(2+L)) shifts storage_in and theta by +N):
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

    fn make_hydro_pos() -> HashMap<EntityId, usize> {
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

    fn make_thermal_pos() -> HashMap<EntityId, usize> {
        // Thermals with EntityId 5 and 6 at positions 0 and 1
        [(EntityId(5), 0), (EntityId(6), 1)].into_iter().collect()
    }

    fn make_bus_pos() -> HashMap<EntityId, usize> {
        // Buses with EntityId 100, 200 at positions 0, 1
        [(EntityId(100), 0), (EntityId(200), 1)]
            .into_iter()
            .collect()
    }

    fn make_line_pos() -> HashMap<EntityId, usize> {
        // Line with EntityId 50 at position 0
        [(EntityId(50), 0)].into_iter().collect()
    }

    fn call(
        var_ref: VariableRef,
        block_idx: usize,
        indexer: &StageIndexer,
        production_models: &ProductionModelSet,
        hydro_pos: &HashMap<EntityId, usize>,
        thermal_pos: &HashMap<EntityId, usize>,
        bus_pos: &HashMap<EntityId, usize>,
        line_pos: &HashMap<EntityId, usize>,
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
        hydro_pos: &HashMap<EntityId, usize>,
        thermal_pos: &HashMap<EntityId, usize>,
        bus_pos: &HashMap<EntityId, usize>,
        line_pos: &HashMap<EntityId, usize>,
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
        resolve_variable_ref(
            &var_ref,
            block_idx,
            indexer.n_blks,
            0, // stage_idx = 0
            indexer,
            production_models,
            &positions,
            &cascade_refs,
        )
    }

    // ── ThermalGeneration tests ───────────────────────────────────────────────

    /// ThermalGeneration block_id=None at block 1 of 3.
    ///
    /// thermal.start = 49, thermal_pos[5] = 0, n_blks = 3, block_idx = 1
    /// Expected column = 49 + 0 * 3 + 1 = 50
    #[test]
    fn thermal_generation_block_id_none_at_block_1() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: None,
            },
            1, // block_idx
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        // thermal.start = 49, pos_5 = 0, n_blks = 3, block = 1
        assert_eq!(result, vec![(49 + 0 * 3 + 1, 1.0)]);
    }

    /// ThermalGeneration with block_id=Some(2) at block 2: should use the explicit block.
    ///
    /// thermal.start = 49, thermal_pos[5] = 0, n_blks = 3, block_id = Some(2)
    /// Expected column = 49 + 0 * 3 + 2 = 51
    #[test]
    fn thermal_generation_block_id_some_at_block_2() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(5),
                block_id: Some(2),
            },
            2,
            &indexer,
            &prod,
            &hpos,
            &tpos,
            &bpos,
            &lpos,
        );

        assert_eq!(result, vec![(49 + 0 * 3 + 2, 1.0)]);
    }

    /// ThermalGeneration for thermal at position 1.
    ///
    /// thermal.start = 49, thermal_pos[6] = 1, n_blks = 3, block = 0
    /// Expected column = 49 + 1 * 3 + 0 = 52
    #[test]
    fn thermal_generation_second_thermal() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::ThermalGeneration {
                thermal_id: EntityId(6),
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

        assert_eq!(result, vec![(49 + 1 * 3 + 0, 1.0)]);
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

        let hpos: HashMap<EntityId, usize> =
            [(EntityId(10), 0), (EntityId(20), 1)].into_iter().collect();
        let tpos: HashMap<EntityId, usize> = HashMap::new();
        let bpos: HashMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
        let lpos: HashMap<EntityId, usize> = HashMap::new();

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
        let result = resolve_variable_ref(
            &VariableRef::HydroEvaporation {
                hydro_id: EntityId(10),
            },
            0,
            1, // n_blks
            0, // stage_idx
            &evap_indexer,
            &prod_models,
            &positions,
            &cascade_refs,
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

        let hpos: HashMap<EntityId, usize> =
            [(EntityId(10), 0), (EntityId(20), 1)].into_iter().collect();
        let tpos: HashMap<EntityId, usize> = HashMap::new();
        let bpos: HashMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
        let lpos: HashMap<EntityId, usize> = HashMap::new();

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
        let result = resolve_variable_ref(
            &VariableRef::HydroEvaporation {
                hydro_id: EntityId(20),
            },
            0,
            1,
            0,
            &evap_indexer,
            &prod_models,
            &positions,
            &cascade_refs,
        );

        assert!(result.is_empty());
    }

    // ── Stub entity tests ─────────────────────────────────────────────────────

    /// PumpingFlow returns empty vec.
    #[test]
    fn pumping_flow_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::PumpingFlow {
                station_id: EntityId(1),
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

    /// PumpingPower returns empty vec.
    #[test]
    fn pumping_power_returns_empty() {
        let indexer = make_indexer();
        let prod = make_production_models();
        let hpos = make_hydro_pos();
        let tpos = make_thermal_pos();
        let bpos = make_bus_pos();
        let lpos = make_line_pos();

        let result = call(
            VariableRef::PumpingPower {
                station_id: EntityId(1),
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
        let hpos: HashMap<EntityId, usize> = HashMap::new();
        let tpos: HashMap<EntityId, usize> =
            [(EntityId(5), 0), (EntityId(6), 1)].into_iter().collect();
        let bpos: HashMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
        let lpos: HashMap<EntityId, usize> = HashMap::new();

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
        let hpos: HashMap<EntityId, usize> = HashMap::new();
        let tpos: HashMap<EntityId, usize> =
            [(EntityId(5), 0), (EntityId(6), 1)].into_iter().collect();
        let bpos: HashMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
        let lpos: HashMap<EntityId, usize> = HashMap::new();

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
        let hpos: HashMap<EntityId, usize> = HashMap::new();
        let tpos: HashMap<EntityId, usize> =
            [(EntityId(5), 0), (EntityId(6), 1)].into_iter().collect();
        let bpos: HashMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
        let lpos: HashMap<EntityId, usize> = HashMap::new();

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
        let hpos: HashMap<EntityId, usize> = HashMap::new();
        let tpos: HashMap<EntityId, usize> =
            [(EntityId(5), 0), (EntityId(6), 1)].into_iter().collect();
        let bpos: HashMap<EntityId, usize> = [(EntityId(100), 0)].into_iter().collect();
        let lpos: HashMap<EntityId, usize> = HashMap::new();

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
        let hpos: HashMap<EntityId, usize> = HashMap::new();
        let tpos: HashMap<EntityId, usize> = HashMap::new();
        let bpos: HashMap<EntityId, usize> = HashMap::new();
        let lpos: HashMap<EntityId, usize> = HashMap::new();

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
