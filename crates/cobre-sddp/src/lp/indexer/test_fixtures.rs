//! Shared `EquipmentCounts` / `FphaColumnLayout` / `EvapConfig` builders for the
//! sibling indexer unit tests.
//!
//! Single owner of the `eq` / `eq_with_anticipated` / `fpha` / `evap` fixtures
//! that the `constructors` test module and downstream `test-support` consumers
//! use. The named `eq` / `eq_with_anticipated` builders set
//! `max_deficit_segments: 1` (a non-degenerate deficit stride), which is **not**
//! the `EquipmentCounts::default()` value `0`; use `..Default::default()`
//! directly when an all-zero count is wanted instead.
//!
//! Compiled under `#[cfg(any(test, feature = "test-support"))]` so plain
//! `cargo test` and downstream integration tests (via `test-support`) both
//! reach the same builders.

use super::layout::{EquipmentCounts, EvapConfig, FphaColumnLayout, StageIndexer};
use super::state_layout::StateLayout;
use super::study_dimensions::StudyDimensions;

/// Build `EquipmentCounts` with the seven scalar entity counts set and no
/// anticipated thermals.
#[must_use]
pub fn eq(
    hydro_count: usize,
    max_par_order: usize,
    n_thermals: usize,
    n_lines: usize,
    n_buses: usize,
    n_blks: usize,
    has_inflow_penalty: bool,
) -> EquipmentCounts {
    EquipmentCounts {
        hydro_count,
        max_par_order,
        n_thermals,
        n_lines,
        n_buses,
        n_blks,
        has_inflow_penalty,
        max_deficit_segments: 1,
        n_anticipated: 0,
        k_max: 0,
        anticipated_lead_stages: vec![],
        anticipated_thermal_indices: vec![],
        n_pumping: 0,
    }
}

/// Build an `FphaColumnLayout` from per-hydro indices and plane counts.
#[must_use]
pub fn fpha(hydro_indices: Vec<usize>, planes_per_hydro: Vec<usize>) -> FphaColumnLayout {
    FphaColumnLayout {
        hydro_indices,
        planes_per_hydro,
    }
}

/// Build an `EvapConfig` covering the given hydro indices.
#[must_use]
pub fn evap(hydro_indices: Vec<usize>) -> EvapConfig {
    EvapConfig { hydro_indices }
}

/// Build a role-(b) [`StageIndexer`] geometry descriptor with **no equipment**,
/// matching the empty-equipment geometry the pre-slim test-only `StageIndexer::new`
/// produced.
///
/// The descriptor carries the role-(b) slots a `TrainingContext.indexer` /
/// `StageExtractionSpec.indexer` consumer needs (`z_inflow_row_start`, `has_ncs`,
/// `max_deficit_segments`) with every equipment / slack / withdrawal range empty
/// (`has_withdrawal == false`, `has_operational_violations == false`). The
/// state-region columns the study implies are owned by the separate
/// [`state_layout`] handle, so the `_hydro_count` / `_max_par_order` arguments are
/// accepted for call-site symmetry but do not add equipment columns here — the
/// geometry is empty exactly as `new` left it.
#[must_use]
pub fn geom(_hydro_count: usize, _max_par_order: usize) -> StageIndexer {
    StageIndexer::with_equipment_and_evaporation(
        &eq(0, 0, 0, 0, 0, 0, false),
        &fpha(vec![], vec![]),
        &evap(vec![]),
    )
}

/// Build a finalized storage+lag [`StateLayout`] (no anticipated thermals) with
/// the full `max_par_order` lag stride for every hydro.
///
/// This is the dense coverage production `build_wired_indexer` finalizes for a
/// study with no per-hydro AR-order truncation, so the layout's
/// `nonzero_state_indices` and `state_to_lp_column_map` caches match a production
/// storage+lag study. The state-fixing patch and cut-path tests read only the
/// state-region column ranges, which are pure functions of `(hydro_count,
/// max_par_order)`.
#[must_use]
pub fn state_layout(hydro_count: usize, max_par_order: usize) -> StateLayout {
    let effective_lag_count = vec![max_par_order; hydro_count];
    StateLayout::new(
        hydro_count,
        max_par_order,
        0,
        0,
        vec![],
        &effective_lag_count,
    )
}

/// Build a finalized [`StateLayout`] from explicit state-vector dimensions,
/// including anticipated thermals.
///
/// `effective_lag_count` is set to the full `max_par_order` for every hydro
/// (dense coverage). `anticipated_lead_stages` must have length `n_anticipated`
/// and its max (when non-empty) must equal `k_max`.
#[must_use]
pub fn state_layout_full(
    hydro_count: usize,
    max_par_order: usize,
    n_anticipated: usize,
    k_max: usize,
    anticipated_lead_stages: Vec<usize>,
) -> StateLayout {
    let effective_lag_count = vec![max_par_order; hydro_count];
    StateLayout::new(
        hydro_count,
        max_par_order,
        n_anticipated,
        k_max,
        anticipated_lead_stages,
        &effective_lag_count,
    )
}

/// Build the role-(a) [`StateLayout`] matching a role-(b) [`StageIndexer`]'s
/// implied state dimensions, recovered from its equipment geometry.
///
/// The slim role-(b) descriptor no longer stores the state-vector dimensions, so
/// they are recovered: `hydro_count` from the per-block stride of the `turbine`
/// range (`turbine.len() / n_blks`), `max_par_order` from the descriptor's
/// `theta`-anchored control-region start. Test sites that already know the
/// dimensions should prefer [`state_layout`] / [`state_layout_full`]; this bridge
/// exists for sites that hold only a built indexer.
///
/// Anticipated thermals are recovered from the `anticipated_decision` column
/// range (its length = `n_anticipated`); the per-plant lead stages are NOT
/// stored on the descriptor, so this bridge assumes none and is valid only for
/// non-anticipated indexers. Anticipated test sites must use
/// [`state_layout_full`].
#[must_use]
pub fn state_layout_for(indexer: &StageIndexer) -> StateLayout {
    debug_assert!(
        indexer.anticipated_decision.is_empty(),
        "state_layout_for assumes no anticipated thermals; use state_layout_full instead"
    );
    let n_blks = indexer.n_blks.max(1);
    let hydro_count = indexer.turbine.len() / n_blks;
    // control_region_start == theta + 1 == turbine.start; theta == N*(3+L) for a
    // non-anticipated layout, so L == theta / N − 3 when N > 0.
    let theta = indexer.turbine.start.saturating_sub(1);
    let max_par_order = if hydro_count > 0 {
        (theta / hydro_count).saturating_sub(3)
    } else {
        0
    };
    state_layout(hydro_count, max_par_order)
}

/// Build an all-default [`StudyDimensions`] (every count `0`, every flag
/// `false`, empty anticipated list).
///
/// Matches the empty non-state shape that the [`geom`] fixture's empty-equipment
/// [`StageIndexer`] implies, for the common test site that builds a `geom`
/// indexer. Sites whose indexer carries real equipment should use
/// [`study_dims_for`] so the non-state facts stay aligned with the indexer.
#[must_use]
pub fn study_dims() -> StudyDimensions {
    StudyDimensions::default()
}

/// Build the [`StudyDimensions`] matching the [`EquipmentCounts`] a test built
/// its [`StageIndexer`] from, so the bridged `study_dims` carries the identical
/// non-state facts the production `build_wired_indexer` derives from the same
/// counts bag.
///
/// The non-state scalars come straight off `counts`; the presence flags use the
/// same predicates the role-(b) constructor applies (`has_inflow_penalty` is the
/// counts flag, `has_withdrawal == hydro_count > 0`,
/// `has_operational_violations == hydro_count != 0`); the anticipated identity
/// list is the counts input the constructor clones. `n_pumping` is `0` on the
/// non-state shape these fixtures imply.
///
/// `has_ncs` is `false`: NCS presence is set only by the production NCS wiring
/// (`!ncs_col_starts.is_empty()`), never by `EquipmentCounts` or by any test
/// fixture — every fixture-built indexer is NCS-inactive, so the constant
/// matches what the deleted `StageIndexer::has_ncs` carried at every bridge site.
#[must_use]
pub fn study_dims_for(counts: &EquipmentCounts) -> StudyDimensions {
    StudyDimensions {
        n_thermals: counts.n_thermals,
        n_lines: counts.n_lines,
        n_buses: counts.n_buses,
        max_deficit_segments: counts.max_deficit_segments,
        has_ncs: false,
        has_inflow_penalty: counts.has_inflow_penalty,
        has_withdrawal: counts.hydro_count > 0,
        has_operational_violations: counts.hydro_count != 0,
        anticipated_thermal_indices: counts.anticipated_thermal_indices.clone(),
        n_pumping: counts.n_pumping,
    }
}

/// Test helper: build `EquipmentCounts` with explicit anticipated thermal
/// fields.
#[must_use]
pub fn eq_with_anticipated(
    hydro_count: usize,
    max_par_order: usize,
    n_thermals: usize,
    n_lines: usize,
    n_buses: usize,
    n_blks: usize,
    has_inflow_penalty: bool,
    n_anticipated: usize,
    k_max: usize,
) -> EquipmentCounts {
    // Default the per-plant K_i array to a uniform `k_max` of length
    // `n_anticipated` so debug asserts on per-plant lead-stage
    // bookkeeping hold. Tests that need a mixed K_i array must construct
    // `EquipmentCounts` directly.
    let anticipated_lead_stages = vec![k_max; n_anticipated];
    let anticipated_thermal_indices = (0..n_anticipated).collect();
    EquipmentCounts {
        hydro_count,
        max_par_order,
        n_thermals,
        n_lines,
        n_buses,
        n_blks,
        has_inflow_penalty,
        max_deficit_segments: 1,
        n_anticipated,
        k_max,
        anticipated_lead_stages,
        anticipated_thermal_indices,
        n_pumping: 0,
    }
}
