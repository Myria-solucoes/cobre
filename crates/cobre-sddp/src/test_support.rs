//! Crate-internal test-support builders: role-(a) [`StateLayout`], role-(b)
//! [`StageGeometry`], and [`StudyDimensions`] fixtures shared across the crate's
//! unit tests and `tests/` integration suites.
//!
//! [`geometry`] drives the production `StageLayout::new`/`StageLayout::geometry`
//! constructors from explicit equipment dimensions, so a test exercises the exact
//! construction path a study of those dimensions would, without a full
//! `StudySetup`. They construct crate-internal types, so they live in `src/` under
//! `#[cfg(any(test, feature = "test-support"))]` — reachable by plain `cargo test`
//! and by downstream integration tests via the `test-support` feature.

use std::collections::{BTreeMap, HashMap};

use chrono::NaiveDate;
use cobre_core::{
    Block, BlockMode, Bus, CascadeTopology, DeficitSegment, EntityId, Hydro, HydroGenerationModel,
    HydroPenalties, NoiseMethod, ResolvedBounds, ResolvedExchangeFactors,
    ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors,
    ResolvedPenalties, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
};
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::hydro_models::{
    EvaporationModel, EvaporationModelSet, FphaPlane, ProductionModelSet, ResolvedProductionModel,
};
use crate::indexer::{CutStateProjection, StateLayout, StudyDimensions};
use crate::lead_time::AnticipatedResolution;
use crate::lp_builder::{ResolvedTables, StageGeometry, StageLayout, TemplateBuildCtx};
use crate::resolved_parameters::ResolvedParameters;
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

/// All-zero [`HydroPenalties`] for [`geometry_hydro`] — no fixture-side penalty
/// cost reaches the column/objective arithmetic `StageLayout::new` computes.
fn geometry_zero_penalties() -> HydroPenalties {
    HydroPenalties {
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
        inflow_nonnegativity_cost: 0.0,
    }
}

/// Fixture hydro at system position `idx`: always `Operating` (`filling`,
/// `entry_stage_id`, `exit_stage_id` all `None`), so `StageLayout::new`'s FPHA/
/// evaporation membership filters never drop a caller-requested index regardless
/// of `stage.id`.
fn geometry_hydro(idx: usize) -> Hydro {
    let id = EntityId(i32::try_from(idx).unwrap_or(i32::MAX));
    Hydro {
        id,
        name: String::new(),
        operational_start_date: NaiveDate::default(),
        bus_id: id,
        downstream_id: None,
        travel_time_hours: None,
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
        penalties: geometry_zero_penalties(),
    }
}

/// Fixture bus at system position `idx` carrying exactly `max_deficit_segments`
/// deficit segments — `StageLayout::new` derives its own `max_deficit_segments` as
/// `ctx.buses.iter().map(|b| b.deficit_segments.len()).max()`, so every bus must
/// carry the caller's count for that derivation to reproduce it.
fn geometry_bus(idx: usize, max_deficit_segments: usize) -> Bus {
    Bus {
        id: EntityId(i32::try_from(idx).unwrap_or(i32::MAX)),
        name: String::new(),
        operational_start_date: NaiveDate::default(),
        deficit_segments: vec![
            DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 0.0,
            };
            max_deficit_segments
        ],
        excess_cost: 0.0,
    }
}

/// Single-stage [`ProductionModelSet`]: `Fpha` with `fpha_planes[local]` planes at
/// each `fpha_hydro_indices[local]`, `ConstantProductivity` elsewhere — the exact
/// classification `StageLayout::new`'s FPHA-membership filter reconstructs from
/// `(hydro, stage)`.
fn geometry_production_models(
    hydro_count: usize,
    fpha_hydro_indices: &[usize],
    fpha_planes: &[usize],
) -> ProductionModelSet {
    let mut plane_count: Vec<Option<usize>> = vec![None; hydro_count];
    for (&h, &planes) in fpha_hydro_indices.iter().zip(fpha_planes) {
        if let Some(slot) = plane_count.get_mut(h) {
            *slot = Some(planes);
        }
    }
    let models = plane_count
        .into_iter()
        .map(|planes| {
            vec![planes.map_or(
                ResolvedProductionModel::ConstantProductivity { productivity: 0.0 },
                |n| ResolvedProductionModel::Fpha {
                    planes: vec![
                        FphaPlane {
                            intercept: 0.0,
                            gamma_v: 0.0,
                            gamma_q: 0.0,
                            gamma_s: 0.0,
                        };
                        n
                    ],
                },
            )]
        })
        .collect();
    ProductionModelSet::new(models, hydro_count, 1)
}

/// Single-hydro [`EvaporationModelSet`]: `Linearized` (membership only — no field
/// beyond variant identity reaches `StageLayout::new`) at each `evap_hydro_indices`
/// position, `None` elsewhere.
fn geometry_evaporation_models(
    hydro_count: usize,
    evap_hydro_indices: &[usize],
) -> EvaporationModelSet {
    let mut is_evap = vec![false; hydro_count];
    for &h in evap_hydro_indices {
        if let Some(slot) = is_evap.get_mut(h) {
            *slot = true;
        }
    }
    let models = is_evap
        .into_iter()
        .map(|evap| {
            if evap {
                EvaporationModel::Linearized {
                    coefficients: Vec::new(),
                    reference_volumes_hm3: Vec::new(),
                }
            } else {
                EvaporationModel::None
            }
        })
        .collect();
    EvaporationModelSet::new(models)
}

/// Single-stage [`Stage`] fixture: `n_blks` uniform blocks, [`BlockMode::Parallel`].
fn geometry_stage(n_blks: usize) -> Stage {
    Stage {
        index: 0,
        id: 0,
        start_date: NaiveDate::default(),
        end_date: NaiveDate::default(),
        season_id: Some(0),
        blocks: (0..n_blks)
            .map(|i| Block {
                index: i,
                name: String::new(),
                duration_hours: 744.0,
            })
            .collect(),
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

/// Build the role-(b) [`StageGeometry`] for a single stage from explicit
/// equipment dimensions, FPHA plane counts, and evaporation hydro indices.
///
/// `fpha_hydro_indices` / `fpha_planes` are parallel (equal length). Builds the
/// production `TemplateBuildCtx`/[`StateLayout`]/[`Stage`] the dimensions
/// describe and delegates to `StageLayout::new`/`StageLayout::geometry` — the
/// single owner of the offset arithmetic.
#[must_use]
// Rationale: `fpha_hydro_indices`/`evap_hydro_indices` stay owned `Vec<usize>` —
// the signature is a stability contract its ~30 call sites depend on — even
// though the body only borrows them (`StageLayout::new` re-derives the
// authoritative membership from `ctx.hydros`/`production_models`/
// `evaporation_models`, not from the caller's raw list).
#[allow(clippy::needless_pass_by_value)]
// Rationale: clippy::similar_names flags `state` next to `stage`; both names
// are established (the `StageLayout`/`StageData` field is `state`, the
// per-stage input is `stage`), so renaming either would obscure intent rather
// than clarify it — mirrors `build_single_stage_template`.
#[allow(clippy::similar_names)]
pub fn geometry(
    dims: &GeometryDims,
    fpha_hydro_indices: Vec<usize>,
    fpha_planes: &[usize],
    evap_hydro_indices: Vec<usize>,
) -> StageGeometry {
    let hydros: Vec<Hydro> = (0..dims.hydro_count).map(geometry_hydro).collect();
    let buses: Vec<Bus> = (0..dims.n_buses)
        .map(|idx| geometry_bus(idx, dims.max_deficit_segments))
        .collect();
    let production_models =
        geometry_production_models(dims.hydro_count, &fpha_hydro_indices, fpha_planes);
    let evaporation_models = geometry_evaporation_models(dims.hydro_count, &evap_hydro_indices);

    let bounds = ResolvedBounds::empty();
    let penalties = ResolvedPenalties::empty();
    let resolved_generic_bounds = ResolvedGenericConstraintBounds::empty();
    let resolved_load_factors = ResolvedLoadFactors::empty();
    let resolved_exchange_factors = ResolvedExchangeFactors::empty();
    let resolved_ncs_bounds = ResolvedNcsBounds::empty();
    let resolved_ncs_factors = ResolvedNcsFactors::empty();
    let resolved_parameters = ResolvedParameters {
        per_param: vec![],
        id_to_slot: vec![],
    };
    let cascade = CascadeTopology::build(&[]);
    let par_lp = PrecomputedPar::default();
    let anticipated_lead_stages = vec![dims.k_max; dims.n_anticipated];

    let ctx = TemplateBuildCtx {
        hydros: &hydros,
        thermals: &[],
        lines: &[],
        buses: &buses,
        load_models: &[],
        cascade: &cascade,
        resolved: ResolvedTables {
            bounds: &bounds,
            penalties: &penalties,
            resolved_generic_bounds: &resolved_generic_bounds,
            resolved_load_factors: &resolved_load_factors,
            resolved_exchange_factors: &resolved_exchange_factors,
            resolved_ncs_bounds: &resolved_ncs_bounds,
            resolved_ncs_factors: &resolved_ncs_factors,
            resolved_parameters: &resolved_parameters,
        },
        hydro_pos: BTreeMap::new(),
        thermal_pos: BTreeMap::new(),
        line_pos: BTreeMap::new(),
        bus_pos: BTreeMap::new(),
        par_lp: &par_lp,
        production_models: &production_models,
        evaporation_models: &evaporation_models,
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
        n_hydros: dims.hydro_count,
        n_thermals: dims.n_thermals,
        n_lines: dims.n_lines,
        n_buses: dims.n_buses,
        max_par_order: dims.max_par_order,
        n_anticipated: dims.n_anticipated,
        k_max: dims.k_max,
        anticipated_lead_stages: anticipated_lead_stages.clone(),
        anticipated_thermal_indices: dims.anticipated_thermal_indices.clone(),
        anticipated_windows: vec![(None, None); dims.n_anticipated],
        anticipated_resolution: AnticipatedResolution::default(),
        study_stage_ids: Vec::new(),
        has_penalty: dims.has_inflow_penalty,
        cumulative_discount_factors: vec![1.0],
        total_hours_per_stage: vec![744.0],
        filling_v_target: BTreeMap::new(),
        arc_stage_weights: HashMap::new(),
        arc_spread_chrono: HashMap::new(),
        arc_arrival_density: HashMap::new(),
        per_stage_mask: Vec::new(),
    };

    let state = state_layout_full(
        dims.hydro_count,
        dims.max_par_order,
        dims.n_anticipated,
        dims.k_max,
        anticipated_lead_stages,
    );
    let stage = geometry_stage(dims.n_blks);

    StageLayout::new(&ctx, &state, &stage, 0).geometry(BlockMode::Parallel)
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
    (0..n_stages)
        .map(|_| cut_state_projection(global))
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
