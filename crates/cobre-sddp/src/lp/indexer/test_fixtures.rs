//! Shared role-(a) [`StateLayout`], role-(b) [`StageGeometry`], and
//! [`StudyDimensions`] builders for the indexer / extraction / patch unit tests.
//!
//! These reproduce the production stage-0 geometry arithmetic from explicit
//! equipment dimensions so a test can build the exact `StageGeometry`,
//! `StateLayout`, and `StudyDimensions` a study with those dimensions would
//! produce, without constructing a full `StudySetup`.
//!
//! Compiled under `#[cfg(any(test, feature = "test-support"))]` so plain
//! `cargo test` and downstream integration tests (via `test-support`) both
//! reach the same builders.

use crate::lp_builder::{
    EVAP_COLS_PER_HYDRO, EVAP_F_MINUS_OFFSET, EVAP_F_PLUS_OFFSET, EVAP_FLOW_OFFSET, StageGeometry,
};

use super::cut_state_projection::CutStateProjection;
use super::layout::EvaporationIndices;
use super::state_layout::StateLayout;
use super::study_dimensions::StudyDimensions;

/// Equipment dimensions for the [`geometry`] / [`study_dims_for`] test builders.
///
/// Mirrors the entity-count inputs a study's stage-0 LP layout is built from.
/// `Default` yields all-zero counts with `max_deficit_segments == 1` (a
/// non-degenerate deficit stride), matching what the named [`eq`] /
/// [`eq_with_anticipated`] builders set.
#[derive(Debug, Clone)]
pub struct GeometryDims {
    /// Number of hydro plants.
    pub hydro_count: usize,
    /// Maximum PAR model order across all hydros.
    pub max_par_order: usize,
    /// Number of thermal units.
    pub n_thermals: usize,
    /// Number of transmission lines.
    pub n_lines: usize,
    /// Number of buses.
    pub n_buses: usize,
    /// Number of demand blocks in the stage.
    pub n_blks: usize,
    /// Whether to include inflow penalty slack columns.
    pub has_inflow_penalty: bool,
    /// Maximum number of deficit segments across all buses.
    pub max_deficit_segments: usize,
    /// Number of anticipated thermals.
    pub n_anticipated: usize,
    /// Maximum `lead_stages` across the anticipated thermals.
    pub k_max: usize,
    /// Mapping from anticipated-local position to global thermal index.
    pub anticipated_thermal_indices: Vec<usize>,
}

impl Default for GeometryDims {
    fn default() -> Self {
        Self {
            hydro_count: 0,
            max_par_order: 0,
            n_thermals: 0,
            n_lines: 0,
            n_buses: 0,
            n_blks: 0,
            has_inflow_penalty: false,
            max_deficit_segments: 1,
            n_anticipated: 0,
            k_max: 0,
            anticipated_thermal_indices: Vec::new(),
        }
    }
}

/// Build [`GeometryDims`] with the seven scalar entity counts set and no
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
) -> GeometryDims {
    GeometryDims {
        hydro_count,
        max_par_order,
        n_thermals,
        n_lines,
        n_buses,
        n_blks,
        has_inflow_penalty,
        ..Default::default()
    }
}

/// Build [`GeometryDims`] with explicit anticipated-thermal fields.
///
/// The anticipated identity list defaults to `0..n_anticipated`.
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
) -> GeometryDims {
    GeometryDims {
        hydro_count,
        max_par_order,
        n_thermals,
        n_lines,
        n_buses,
        n_blks,
        has_inflow_penalty,
        n_anticipated,
        k_max,
        anticipated_thermal_indices: (0..n_anticipated).collect(),
        ..Default::default()
    }
}

/// Build the role-(b) [`StageGeometry`] for a single stage from explicit
/// equipment dimensions, FPHA plane counts, and evaporation hydro indices.
///
/// Reproduces the production stage-0 column/row arithmetic
/// (`build_single_stage_template` / `StageLayout`): `theta = N*(3+L) + A*k_max + A`
/// anchors the control region at `theta + 1`, equipment columns follow in the
/// canonical order (turbine, spillage, diversion, thermal, `anticipated_decision`,
/// lines, deficit, excess, `inflow_slack`, generation, evaporation, withdrawal
/// slacks, operational-violation slacks), and rows follow z-inflow → water
/// balance → load balance → FPHA → evaporation. The returned geometry is correct
/// for a single stage whose block count is `dims.n_blks`.
///
/// `fpha_hydro_indices` / `fpha_planes` describe the FPHA hydros at this stage
/// (parallel, equal length); `evap_hydro_indices` lists the evaporation hydros.
#[must_use]
// Rationale: single cohesive LP column/row layout reproduction; every local
// binding contributes to the `StageGeometry { .. }` literal that closes the
// function. Splitting into sub-helpers would scatter the offset derivation order
// and obscure the one-shot build contract where each offset derives from the
// previous.
#[allow(clippy::too_many_lines)]
pub fn geometry(
    dims: &GeometryDims,
    fpha_hydro_indices: Vec<usize>,
    fpha_planes: &[usize],
    evap_hydro_indices: Vec<usize>,
) -> StageGeometry {
    let GeometryDims {
        hydro_count,
        max_par_order,
        n_thermals,
        n_lines,
        n_buses,
        n_blks,
        has_inflow_penalty,
        max_deficit_segments,
        n_anticipated,
        k_max,
        ..
    } = *dims;
    let n_ant_state = n_anticipated * k_max;

    let theta = hydro_count * (3 + max_par_order) + n_ant_state + n_anticipated;
    let decision_start = theta + 1;

    let z_inflow_row_start = 0_usize;

    let turbine_start = decision_start;
    let spillage_start = turbine_start + hydro_count * n_blks;
    let diversion_start = spillage_start + hydro_count * n_blks;
    let thermal_start = diversion_start + hydro_count * n_blks;
    let thermal_end = thermal_start + n_thermals * n_blks;
    let anticipated_decision = if n_anticipated > 0 {
        thermal_end..thermal_end + n_anticipated
    } else {
        0..0
    };
    let line_fwd_start = thermal_end + n_anticipated;
    let line_rev_start = line_fwd_start + n_lines * n_blks;
    let deficit_start = line_rev_start + n_lines * n_blks;
    let excess_start = deficit_start + n_buses * max_deficit_segments * n_blks;
    let excess_end = excess_start + n_buses * n_blks;

    let (inflow_slack, active_penalty) = if has_inflow_penalty && hydro_count > 0 {
        (excess_end..excess_end + hydro_count, true)
    } else {
        (0..0, false)
    };

    let n_fpha_hydros = fpha_hydro_indices.len();
    let generation_start = if active_penalty {
        inflow_slack.end
    } else {
        excess_end
    };
    let generation_end = generation_start + n_fpha_hydros * n_blks;
    let generation = if n_fpha_hydros > 0 {
        generation_start..generation_end
    } else {
        0..0
    };

    let n_evap_hydros = evap_hydro_indices.len();
    let evap_col_start = generation_end;

    let water_balance_start = z_inflow_row_start + hydro_count;
    let load_balance_start = water_balance_start + hydro_count;
    let load_balance_end = load_balance_start + n_buses * n_blks;

    let mut fpha_row_cursor = load_balance_end;
    for &planes in fpha_planes {
        fpha_row_cursor += planes * n_blks;
    }

    let evap_indices: Vec<EvaporationIndices> = (0..n_evap_hydros)
        .map(|i| EvaporationIndices {
            evaporation_flow_col: evap_col_start + i * EVAP_COLS_PER_HYDRO + EVAP_FLOW_OFFSET,
            f_evap_plus_col: evap_col_start + i * EVAP_COLS_PER_HYDRO + EVAP_F_PLUS_OFFSET,
            f_evap_minus_col: evap_col_start + i * EVAP_COLS_PER_HYDRO + EVAP_F_MINUS_OFFSET,
            evap_row: fpha_row_cursor + i,
        })
        .collect();
    let evap_col_end = evap_col_start + n_evap_hydros * EVAP_COLS_PER_HYDRO;

    let (withdrawal_slack_neg, withdrawal_slack_pos) = if hydro_count > 0 {
        let neg = evap_col_end..evap_col_end + hydro_count;
        let pos = neg.end..neg.end + hydro_count;
        (neg, pos)
    } else {
        (0..0, 0..0)
    };

    let ws_end = withdrawal_slack_pos.end;
    let (outflow_below_slack, outflow_above_slack, turbine_below_slack, generation_below_slack) =
        if hydro_count == 0 {
            (0..0, 0..0, 0..0, 0..0)
        } else {
            let n_op = hydro_count * n_blks;
            let ob = ws_end..ws_end + n_op;
            let oa = ob.end..ob.end + n_op;
            let tb = oa.end..oa.end + n_op;
            let gb = tb.end..tb.end + n_op;
            (ob, oa, tb, gb)
        };

    StageGeometry {
        // θ sits one column before the turbine block (`turbine.start == theta + 1`).
        theta_col: turbine_start - 1,
        turbine: turbine_start..spillage_start,
        spillage: spillage_start..diversion_start,
        diversion: diversion_start..thermal_start,
        thermal: thermal_start..thermal_end,
        anticipated_decision,
        line_fwd: line_fwd_start..line_rev_start,
        line_rev: line_rev_start..deficit_start,
        deficit: deficit_start..excess_start,
        excess: excess_start..excess_end,
        generation,
        evap_indices,
        inflow_slack,
        withdrawal_slack_neg,
        withdrawal_slack_pos,
        outflow_below_slack,
        outflow_above_slack,
        turbine_below_slack,
        generation_below_slack,
        // This fixture models no contract columns; the production
        // `start..start`-at-pumping-end anchoring is owned by
        // `StageGeometry::from_layout`.
        contract_import: 0..0,
        contract_export: 0..0,
        water_balance: water_balance_start..water_balance_start + hydro_count,
        load_balance: load_balance_start..load_balance_end,
        // This fixture models no filling hydros, so the terminal-target and
        // operating-floor blocks are empty.
        filling_target: 0..0,
        filling_target_col: 0..0,
        filled_min_storage_floor: 0..0,
        filled_min_storage_floor_col: 0..0,
        z_inflow_row_start,
        n_blks,
        fpha_hydro_indices,
        evap_hydro_indices,
        filling_target_hydro_indices: vec![],
        filled_min_storage_floor_hydro_indices: vec![],
    }
}

/// Build the empty-equipment role-(b) [`StageGeometry`] (every range `0..0`).
///
/// The `_hydro_count` / `_max_par_order` arguments are accepted for call-site
/// symmetry but add no equipment columns — the state-region columns the study
/// implies are owned by the separate [`StateLayout`] handle.
#[must_use]
pub fn geom(_hydro_count: usize, _max_par_order: usize) -> StageGeometry {
    StageGeometry::default()
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

/// Build the all-enabled per-pool [`CutStateProjection`] vector (one entry per stage)
/// the default training paths use: every pool projects the full global state, so
/// `cut_state_layouts[t].n_state() == global.n_state` for all `t`.
///
/// Tests that drive the backward pass need this slice on `TrainingContext`; the
/// default (all-enabled) projection keeps the extracted subgradient bit-identical
/// to the global-loop result.
#[must_use]
pub fn all_enabled_cut_state_layouts(
    global: &StateLayout,
    n_stages: usize,
) -> Vec<CutStateProjection> {
    let full = cobre_core::temporal::StageStateConfig {
        storage: true,
        inflow_lags: true,
    };
    (0..n_stages)
        .map(|_| CutStateProjection::new(global, full))
        .collect()
}

/// Build a single all-enabled per-pool [`CutStateProjection`] projecting the full
/// global state — the projection a cut-row builder test threads alongside a
/// full-dimension [`StateLayout`] so the render reproduces the global nonzero
/// mask.
#[must_use]
pub fn cut_state_projection(global: &StateLayout) -> CutStateProjection {
    CutStateProjection::new(
        global,
        cobre_core::temporal::StageStateConfig {
            storage: true,
            inflow_lags: true,
        },
    )
}

/// Build an all-default [`StudyDimensions`] (every count `0`, every flag
/// `false`, empty anticipated list).
#[must_use]
pub fn study_dims() -> StudyDimensions {
    StudyDimensions::default()
}

/// Build the [`StudyDimensions`] matching the [`GeometryDims`] a test built its
/// stage geometry from, so the non-state shape stays aligned with the geometry.
///
/// The non-state scalars come straight off `dims`; the presence flags use the
/// production predicates (`has_inflow_penalty` is the flag, `has_withdrawal ==
/// hydro_count > 0`, `has_operational_violations == hydro_count != 0`).
///
/// `has_ncs` is `false`: NCS presence is set only by the production NCS wiring
/// (`!ncs_col_starts.is_empty()`), never by these fixtures — every fixture-built
/// stage is NCS-inactive. `n_pumping` is `0` on the non-state shape these
/// fixtures imply.
#[must_use]
pub fn study_dims_for(dims: &GeometryDims) -> StudyDimensions {
    StudyDimensions {
        n_thermals: dims.n_thermals,
        n_lines: dims.n_lines,
        n_buses: dims.n_buses,
        max_deficit_segments: dims.max_deficit_segments,
        has_ncs: false,
        has_inflow_penalty: dims.has_inflow_penalty,
        has_withdrawal: dims.hydro_count > 0,
        has_operational_violations: dims.hydro_count != 0,
        anticipated_thermal_indices: dims.anticipated_thermal_indices.clone(),
        n_pumping: 0,
    }
}
