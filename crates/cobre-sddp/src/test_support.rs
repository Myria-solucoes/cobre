//! Crate-internal test-support builders: role-(a) [`StateLayout`], role-(b)
//! [`StageGeometry`], and [`StudyDimensions`] fixtures shared across the crate's
//! unit tests and `tests/` integration suites.
//!
//! They reproduce production stage-0 layout arithmetic from explicit equipment
//! dimensions, so a test builds the exact types a study of those dimensions would
//! without a full `StudySetup`. They construct crate-internal types, so they live
//! in `src/` under `#[cfg(any(test, feature = "test-support"))]` — reachable by
//! plain `cargo test` and by downstream integration tests via the `test-support`
//! feature.

use crate::indexer::{CutStateProjection, EvaporationIndices, StateLayout, StudyDimensions};
use crate::lp_builder::{
    EVAP_COLS_PER_HYDRO, EVAP_F_MINUS_OFFSET, EVAP_F_PLUS_OFFSET, EVAP_FLOW_OFFSET, StageGeometry,
};
use cobre_solver::StageTemplate;

/// Equipment dimensions for the [`geometry`] / [`study_dims_for`] test builders.
///
/// `Default` sets `max_deficit_segments == 1` (a non-degenerate deficit stride);
/// every other count is `0`.
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
/// `fpha_hydro_indices` / `fpha_planes` are parallel (equal length).
#[must_use]
// Rationale: each offset derives from the previous, so splitting into sub-helpers
// would scatter the sequential layout derivation.
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

    // `2 * n_ant_state`: the anticipated ring's outgoing + incoming blocks.
    let theta = hydro_count * (3 + max_par_order) + 2 * n_ant_state;
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

    let mut evap_indices: Vec<EvaporationIndices> = Vec::with_capacity(n_evap_hydros * n_blks);
    for i in 0..n_evap_hydros {
        for blk in 0..n_blks {
            let slot = i * n_blks + blk;
            let triple_base = evap_col_start + slot * EVAP_COLS_PER_HYDRO;
            evap_indices.push(EvaporationIndices {
                evaporation_flow_col: triple_base + EVAP_FLOW_OFFSET,
                f_evap_plus_col: triple_base + EVAP_F_PLUS_OFFSET,
                f_evap_minus_col: triple_base + EVAP_F_MINUS_OFFSET,
                evap_row: fpha_row_cursor + slot,
            });
        }
    }
    let evap_col_end = evap_col_start + n_evap_hydros * n_blks * EVAP_COLS_PER_HYDRO;

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
        contract_import: 0..0,
        contract_export: 0..0,
        water_balance: water_balance_start..water_balance_start + hydro_count,
        load_balance: load_balance_start..load_balance_end,
        filling_target: 0..0,
        filling_target_col: 0..0,
        filled_min_storage_floor: 0..0,
        filled_min_storage_floor_col: 0..0,
        z_inflow_row_start,
        n_blks,
        storage_internal_start: 0,
        block_mode: cobre_core::BlockMode::Parallel,
        fpha_hydro_indices,
        evap_hydro_indices,
        filling_target_hydro_indices: vec![],
        filled_min_storage_floor_hydro_indices: vec![],
    }
}

/// Build the empty-equipment role-(b) [`StageGeometry`] (every range `0..0`).
///
/// The `_hydro_count` / `_max_par_order` arguments are ignored — accepted only for
/// call-site symmetry with [`geometry`].
#[must_use]
pub fn geom(_hydro_count: usize, _max_par_order: usize) -> StageGeometry {
    StageGeometry::default()
}

/// Build a finalized storage+lag [`StateLayout`] (no anticipated thermals) with the
/// full `max_par_order` lag stride for every hydro — the dense coverage
/// `crate::setup::resolve_state_layout` finalizes with no per-hydro AR truncation.
#[must_use]
pub fn state_layout(hydro_count: usize, max_par_order: usize) -> StateLayout {
    let effective_lag_count = vec![max_par_order; hydro_count];
    StateLayout::new(
        hydro_count,
        max_par_order,
        0,
        Vec::new(),
        0,
        0,
        vec![],
        &effective_lag_count,
    )
}

/// Build a finalized [`StateLayout`] from explicit state-vector dimensions,
/// including anticipated thermals. Lag coverage is dense (full `max_par_order`).
///
/// `anticipated_lead_stages` must have length `n_anticipated` and its max (when
/// non-empty) must equal `k_max`.
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
        0,
        Vec::new(),
        n_anticipated,
        k_max,
        anticipated_lead_stages,
        &effective_lag_count,
    )
}

/// Build a finalized [`StateLayout`] with a declared travel-time bucket block
/// (`transit_buckets_out`/`transit_buckets_in`), optionally combined with anticipated
/// thermals. `effective_lag_count` is dense (full `max_par_order` for every
/// hydro), matching [`state_layout_full`].
#[must_use]
pub fn state_layout_with_transit_buckets(
    hydro_count: usize,
    max_par_order: usize,
    n_buckets: usize,
    transit_bucket_column_order: Vec<(usize, usize)>,
    n_anticipated: usize,
    k_max: usize,
    anticipated_lead_stages: Vec<usize>,
) -> StateLayout {
    let effective_lag_count = vec![max_par_order; hydro_count];
    StateLayout::new(
        hydro_count,
        max_par_order,
        n_buckets,
        transit_bucket_column_order,
        n_anticipated,
        k_max,
        anticipated_lead_stages,
        &effective_lag_count,
    )
}

/// Bucket-only [`StageTemplate`]: `num_cols` free columns, zero rows. `n_hydro = 0`
/// so noise transformation never runs.
#[must_use]
pub fn transit_bucket_only_template(num_cols: usize, n_state: usize) -> StageTemplate {
    StageTemplate {
        num_cols,
        num_rows: 0,
        num_nz: 0,
        col_starts: vec![0_i32; num_cols + 1],
        row_indices: Vec::new(),
        values: Vec::new(),
        col_lower: vec![f64::NEG_INFINITY; num_cols],
        col_upper: vec![f64::INFINITY; num_cols],
        objective: vec![0.0; num_cols],
        row_lower: Vec::new(),
        row_upper: Vec::new(),
        n_state,
        n_transfer: 0,
        n_dual_relevant: 0,
        n_hydro: 0,
        max_par_order: 0,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    }
}

/// Build the all-enabled per-pool [`CutStateProjection`] vector (one per stage): every
/// pool projects the full global state (`n_state() == global.n_state` for all `t`),
/// keeping the extracted subgradient bit-identical to the unprojected global loop.
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

/// Build a single all-enabled [`CutStateProjection`] projecting the full global state.
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
/// stage geometry from. `has_ncs` is always `false`: these fixtures never model NCS
/// (production sets it from `!ncs_col_starts.is_empty()`).
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
