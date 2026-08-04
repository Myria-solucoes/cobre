//! Crate-internal test-support builders: role-(a) [`StateSpace`], role-(b)
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
use cobre_core::scenario::{InflowModel, LoadModel, SamplingScheme};
use cobre_core::temporal::{Node as PolicyNode, PolicyGraphType, Transition};
use cobre_core::{
    Block, BlockMode, BoundsCountsSpec, BoundsDefaults, Bus, BusStagePenalties, CascadeTopology,
    ContractBlockBounds, DeficitSegment, EntityId, Hydro, HydroBlockBounds, HydroGenerationModel,
    HydroPenalties, HydroStageBounds, HydroStagePenalties, HydroStorage, HydroUnitGroup,
    InitialConditions, LineBlockBounds, LineStagePenalties, NcsStagePenalties, NoiseMethod,
    PenaltiesCountsSpec, PenaltiesDefaults, PolicyGraph, PumpingBlockBounds, ResolvedBounds,
    ResolvedGenericConstraintBounds, ResolvedLoadFactors, ResolvedNcsBounds, ResolvedNcsFactors,
    ResolvedPenalties, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig, System,
    SystemBuilder, ThermalBlockBounds, ThermalStageBounds,
};
use cobre_io::StageIdResolver;
use cobre_io::config::{
    Config, EstimationConfig, ExportsConfig, InflowNonNegativityConfig, InflowNonNegativityMethod,
    ModelingConfig, ParallelismConfig, PolicyConfig, RowSelectionConfig,
    SimulationConfig as IoSimulationConfig, SimulationSelection, StateSpaceConfig, StoppingMode,
    StoppingRuleConfig, TrainingConfig, TrainingSelection, TrainingSolverConfig,
    UpperBoundEvaluationConfig,
};
use cobre_stochastic::par::precompute::PrecomputedPar;
use cobre_stochastic::{
    ClassSchemes, OpeningTreeInputs, StochasticContext, build_stochastic_context,
};

use crate::StudySetup;
use crate::context::{StageContext, TrainingContext};
use crate::cut::pool::CutPool;
use crate::error::SddpError;
use crate::hydro_models::{
    EvaporationModel, EvaporationModelSet, FphaPlane, PrepareHydroModelsResult, ProductionModelSet,
    ResolvedProductionModel,
};
use crate::indexer::{CutStateProjection, HydroCellIndex, StateSpace, StudyDimensions, ThermalSys};
use crate::lead_time::AnticipatedResolution;
use crate::lp_builder::{ResolvedTables, StageGeometry, StageLayout, TemplateBuildCtx};
use crate::policy::policy_load::{
    FullFcf, PolicyLoadProof, PolicyStageManifest, validate_policy_load,
};
use crate::resolved_parameters::ResolvedParameters;
use crate::setup::node_graph::{NodeGraph, build_node_graph, enumerated_scenario_count};
use crate::solve::stage_solve::{StageInputs, run_stage_solve};
use crate::training::stage_solve_prep::{
    InflowNoise, LoadNoise, OpeningMode, StageSolvePrep, StageSolvePrepParams, StateSource,
};
use crate::trajectory::TrajectoryRecord;
use crate::workspace::{CapturedBasis, SolverWorkspace};
use cobre_solver::{Basis, BasisStatus, SolutionView, SolverInterface, StageTemplate};

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

/// Overwrite `out` with a basis satisfying `col_basic + row_basic ==
/// row_status.len()` — the consistency invariant every real solver's `get_basis`
/// returns and [`enforce_basic_count_invariant`] requires of a stored basis.
///
/// A mock `get_basis` that leaves `out` untouched keeps the all-`Lower` zero-fill
/// from `Basis::new`, i.e. `total_basic == 0`. No solver produces that, and
/// offering it back as a warm start is a basic-count deficit, which
/// [`enforce_basic_count_invariant`] rejects as a shape mismatch.
///
/// [`enforce_basic_count_invariant`]: crate::basis_reconstruct::enforce_basic_count_invariant
pub fn fill_consistent_basis(out: &mut Basis) {
    let num_row = out.row_status.len();
    out.col_status.fill(BasisStatus::Lower);
    out.row_status.fill(BasisStatus::Lower);
    let basic_cols = num_row.min(out.col_status.len());
    out.col_status[..basic_cols].fill(BasisStatus::Basic);
    out.row_status[..num_row - basic_cols].fill(BasisStatus::Basic);
}

/// Build [`GeometryDims`] with the scalar entity counts set and no anticipated
/// thermals.
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

/// Build a [`HydroUnitGroup`] with the given `id`, `bus_id`, and four bounds —
/// the shared multi-bus fixture helper (`unit_groups` cannot otherwise be
/// populated from outside `cobre-io`'s `pub(super)` validation-module helper).
#[must_use]
pub fn make_unit_group(
    id: EntityId,
    bus_id: EntityId,
    min_generation_mw: f64,
    max_generation_mw: f64,
    min_turbined_m3s: f64,
    max_turbined_m3s: f64,
) -> HydroUnitGroup {
    HydroUnitGroup {
        id,
        name: format!("G{id}"),
        bus_id,
        min_generation_mw,
        max_generation_mw,
        min_turbined_m3s,
        max_turbined_m3s,
    }
}

/// Fixture hydro at system position `idx`: always `Operating` (`filling`,
/// `entry_stage_id`, `exit_stage_id` all `None`), so `StageLayout::new`'s FPHA/
/// evaporation membership filters never drop a caller-requested index regardless
/// of `stage.id`. Declares its mirror unit group before return — `geometry` never
/// mutates the collected `Vec`, so the return is the finalization boundary — and
/// stays `pub(crate)` for `indexer::hydro_cell`'s identity test, which needs these
/// exact hydros.
pub(crate) fn geometry_hydro(idx: usize) -> Hydro {
    let id = EntityId(i32::try_from(idx).unwrap_or(i32::MAX));
    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id,
        name: String::new(),
        operational_start_date: NaiveDate::default(),
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
    };
    hydro.declare_mirror_unit_group(id);
    hydro
}

/// `geometry_hydro` with caller-chosen `unit_groups` and `generation_model`,
/// for multi-bus [`HydroCellIndex`] fixtures — every other field matches
/// `geometry_hydro` exactly, so a single-group caller gets the identical hydro.
#[must_use]
pub fn geometry_hydro_with_groups(
    idx: usize,
    unit_groups: Vec<HydroUnitGroup>,
    generation_model: HydroGenerationModel,
) -> Hydro {
    let mut hydro = geometry_hydro(idx);
    hydro.unit_groups = unit_groups;
    hydro.generation_model = generation_model;
    hydro.declare_mirror_unit_group(EntityId(i32::try_from(idx).unwrap_or(i32::MAX)));
    // Both calls earn their place: the declare covers a caller passing an empty
    // vec, and the sort supplies the id-ascending `unit_groups` that
    // `HydroCellIndex::build` documents as its precondition for keeping
    // `cell_group_pos` ascending within a cell.
    hydro.sort_unit_groups();
    hydro
}

/// Identity [`HydroCellIndex`] for `n_hydros` single-bus hydros
/// (`cells_of(h) == h..h+1` for every `h`) — the single shared builder for every
/// fixture in the crate that needs a `HydroCellIndex` but is not itself testing
/// the multi-bus partition. Safe to over-size: a caller passing more than its
/// fixture's actual hydro count still gets a correct `cells_of` for every hydro
/// it actually queries.
#[must_use]
pub fn identity_hydro_cell_index(n_hydros: usize) -> HydroCellIndex {
    HydroCellIndex::build(&(0..n_hydros).map(geometry_hydro).collect::<Vec<_>>())
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
/// production `TemplateBuildCtx`/[`StateSpace`]/[`Stage`] the dimensions
/// describe and delegates to `StageLayout::new`/`StageLayout::geometry` — the
/// single owner of the offset arithmetic.
#[must_use]
// Rationale: `fpha_hydro_indices`/`evap_hydro_indices` stay owned `Vec<usize>` —
// the signature is a stability contract its call sites depend on — even
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
    let hydro_cell_index = HydroCellIndex::build(&hydros);
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
    let resolved_ncs_bounds = ResolvedNcsBounds::empty();
    let resolved_ncs_factors = ResolvedNcsFactors::empty();
    let resolved_parameters = ResolvedParameters {
        per_param: vec![],
        id_to_slot: vec![],
        cost_scale_factor: 1_000_000.0,
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
        hydro_cell_index: &hydro_cell_index,
        resolved: ResolvedTables {
            bounds: &bounds,
            penalties: &penalties,
            resolved_generic_bounds: &resolved_generic_bounds,
            resolved_load_factors: &resolved_load_factors,
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
        anticipated_thermal_indices: dims
            .anticipated_thermal_indices
            .iter()
            .map(|&t| ThermalSys::new(t))
            .collect(),
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

/// Build a finalized storage+lag [`StateSpace`] (no anticipated thermals) with the
/// full `max_par_order` lag stride for every hydro — the dense coverage
/// `crate::setup::resolve_state_layout` finalizes with no per-hydro AR truncation.
#[must_use]
pub fn state_layout(hydro_count: usize, max_par_order: usize) -> StateSpace {
    state_layout_full(hydro_count, max_par_order, 0, 0, Vec::new())
}

/// Build a finalized [`StateSpace`] from explicit state-vector dimensions,
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
) -> StateSpace {
    state_layout_with_transit_buckets(
        hydro_count,
        max_par_order,
        0,
        Vec::new(),
        n_anticipated,
        k_max,
        anticipated_lead_stages,
    )
}

/// Build a finalized [`StateSpace`] with a declared travel-time bucket block
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
) -> StateSpace {
    let effective_lag_count = vec![max_par_order; hydro_count];
    StateSpace::new(
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
    global: &StateSpace,
    n_stages: usize,
) -> Vec<CutStateProjection> {
    (0..n_stages)
        .map(|_| cut_state_projection(global))
        .collect()
}

/// Build a single all-enabled [`CutStateProjection`] projecting the full global state.
#[must_use]
pub fn cut_state_projection(global: &StateSpace) -> CutStateProjection {
    CutStateProjection::new(
        global,
        StageStateConfig {
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

/// Trivial matching [`PolicyLoadProof`] typed to [`FullFcf`] for tests
/// exercising FCF reconstruction rather than cross-study compatibility: identical
/// `state_dimension`/`num_stages` on both sides with an empty manifest (the
/// "identity could not be verified" warning path). [`validate_policy_load`] is
/// the only constructor of [`PolicyLoadProof`], so tests route through it here
/// rather than a forged literal.
///
/// # Panics
///
/// Never in practice — see the rationale below.
#[allow(clippy::expect_used)]
// Rationale: matching state_dimension/num_stages with an empty manifest on
// both sides cannot hit validate_policy_load's error paths (state_dimension
// and num_stages equality hold trivially; an empty manifest short-circuits
// identity comparison with a warning, never an error).
#[must_use]
pub fn trivial_full_fcf_proof(state_dimension: u32, num_stages: u32) -> PolicyLoadProof<FullFcf> {
    let manifest = PolicyStageManifest {
        state_dimension,
        num_stages,
        slots: &[],
    };
    validate_policy_load::<FullFcf>(&manifest, &manifest)
        .expect("trivial matching manifest cannot fail validate_policy_load")
}

/// Patch one stage-LP solve exactly as the production backward pass's
/// `patch_opening_bounds` does (`training/backward/lp_setup.rs`): delegates
/// verbatim to `StageSolvePrep::run` with the backward-opening variation
/// point (`OpeningMode::PerOpening`, `LoadNoise::Present`,
/// `InflowNoise::Transform`) — no probe-side reimplementation of the
/// patch pipeline (lag-folded water-balance RHS, NCS availability,
/// commitment reconciliation).
///
/// # Errors
///
/// Propagates [`SddpError::AnticipatedCommitmentOutOfBounds`] exactly as
/// `StageSolvePrep::run` does.
pub fn patch_backward_opening_for_probe<S: SolverInterface + Send>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    training_ctx: &TrainingContext<'_>,
    stage: usize,
    pinned_state: &[f64],
    raw_noise: &[f64],
) -> Result<(), SddpError> {
    let prep_params = StageSolvePrepParams {
        state_source: StateSource(pinned_state),
        opening_mode: OpeningMode::PerOpening,
        load_noise: LoadNoise::Present,
        inflow_noise: InflowNoise::Transform,
        raw_noise,
    };
    StageSolvePrep::run(
        &mut ws.solver,
        &mut ws.patch_buf,
        &mut ws.scratch,
        ctx,
        training_ctx,
        stage,
        &prep_params,
    )
}

/// Run one stage-LP solve exactly as production's shared `run_stage_solve`
/// entry point does (`solve::stage_solve`, `pub(crate)` — the module
/// forward/backward/simulation all route through, no external consumer by
/// design): basis reconstruction by cut-pool slot identity when
/// `stored_basis` is `Some`, else an implicit warm start from whatever the
/// solver instance currently retains. Reachable here so a probe can drive the
/// identical hot path on a scratch workspace without duplicating its
/// reconstruction/invariant-enforcement logic.
///
/// # Errors
///
/// Propagates the same [`SddpError`] variants as production's stage solve
/// (`Infeasible`, `Solver`, or a basis-shape mismatch).
pub fn solve_stage_for_probe<'ws, S: SolverInterface>(
    ws: &'ws mut SolverWorkspace<S>,
    stage_context: &StageContext<'_>,
    pool: &CutPool,
    stored_basis: Option<&CapturedBasis>,
    stage_index: usize,
    scenario_index: usize,
    node_id: i32,
) -> Result<SolutionView<'ws>, SddpError> {
    let inputs = StageInputs {
        stage_context,
        pool,
        stored_basis,
        stage_index,
        scenario_index,
        iteration: None,
        node_id,
    };
    run_stage_solve(ws, &inputs)
}

/// Trial-point states in the flat shape the passes index, `records[m * n_stages + stage]`.
///
/// Each scenario's state is replicated across every stage, so the backward pass's
/// per-stage repack of the gather buffers reproduces exactly `states` at whichever
/// stage it runs — letting a test pin cut arithmetic against a known trial point.
#[must_use]
pub fn trial_state_records(states: &[Vec<f64>], n_stages: usize) -> Vec<TrajectoryRecord> {
    states
        .iter()
        .flat_map(|state| {
            (0..n_stages).map(move |_| TrajectoryRecord {
                primal: Vec::new(),
                dual: Vec::new(),
                stage_cost: 0.0,
                node_id: 0,
                state: state.clone(),
            })
        })
        .collect()
}

/// The byte-exact chain [`NodeGraph`] (C1) for `stochastic`: one node per
/// stage, pools 1:1, each node's `Ω` view spanning exactly that stage's
/// [`StochasticContext::opening_tree`] openings. Delegates to
/// [`build_node_graph`]'s own chain path (an empty `nodes[]`) instead of
/// re-deriving the shape, so a fixture's declared branching factor and its
/// node graph can never drift apart.
///
/// # Panics
///
/// Never in practice — see the rationale below.
#[allow(clippy::expect_used)]
// Rationale: build_node_graph only returns Err for a transition/node naming
// an undeclared id; PolicyGraph::default() carries no nodes/transitions at
// all, so that error path is unreachable here.
#[must_use]
pub fn chain_node_graph(stochastic: &StochasticContext) -> NodeGraph {
    let n_stages = stochastic.n_stages();
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let study_stage_ids: Vec<i32> = (0..n_stages as i32).collect();
    let resolver = StageIdResolver::from_study_stage_ids(&study_stage_ids);
    build_node_graph(&PolicyGraph::default(), n_stages, &resolver, stochastic)
        .expect("chain_node_graph: build_node_graph never errors for an empty nodes[] graph")
}

// ── DECOMP K-fan fixture ─────────────────────────────────────────────────

/// Fixed training seed for [`k_fan_setup`] — every caller trains the identical,
/// reproducible sampled walk; not a caller-configurable knob.
const K_FAN_SEED: u64 = 42;
/// [`K_FAN_SEED`], typed for [`Config`]'s `training.tree_seed` field.
const K_FAN_TREE_SEED: i64 = 42;

/// Study-stage id of the K-fan's root (a single node, the sole stage-0 alive node).
const K_FAN_ROOT_STAGE_ID: i32 = 0;
/// Study-stage id of the fan level: `k` distinct nodes, each cut-generating (each
/// owns exactly one leaf successor).
const K_FAN_BRANCH_STAGE_ID: i32 = 1;
/// Study-stage id of the leaf level: `k` distinct nodes sharing ONE pool
/// (`build_node_graph`'s leaf-sharing rule) — never cut-generating.
const K_FAN_LEAF_STAGE_ID: i32 = 2;

/// The declared `nodes[]`/`transitions[]` K-fan: root (id `0`) branches into fan
/// nodes `1..=k` under strictly non-uniform weights `i / Σj` (never a uniform
/// `1/k` split — a uniform split would make every reduction order sum identical
/// terms, defeating the canonical-order gate's power), each fan node `i` then
/// deterministically reaching its own leaf `k+i`. `num_nodes = 2k+1`,
/// `n_pools = k+2` (root + k fan pools + one shared leaf pool) —
/// `num_nodes > n_pools` for every `k >= 2`, so a canonical-node-position-as-pool-id
/// conflation bug would misroute or overflow a pool here, where it cannot on a
/// chain (`node_index == pool_id` there hides the bug).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
)]
fn k_fan_policy_graph(k: usize, reversed: bool) -> PolicyGraph {
    debug_assert!(k >= 2, "k_fan_policy_graph: k must be >= 2 (DECOMP shape)");
    let mut nodes = Vec::with_capacity(1 + 2 * k);
    let mut transitions = Vec::with_capacity(2 * k);
    nodes.push(PolicyNode {
        id: 0,
        stage_id: K_FAN_ROOT_STAGE_ID,
        scenario_id: None,
        label: None,
    });
    let total_weight: f64 = (1..=k).map(|i| i as f64).sum();
    for i in 1..=k {
        let fan_id = i as i32;
        let leaf_id = fan_id + k as i32;
        nodes.push(PolicyNode {
            id: fan_id,
            stage_id: K_FAN_BRANCH_STAGE_ID,
            scenario_id: None,
            label: None,
        });
        nodes.push(PolicyNode {
            id: leaf_id,
            stage_id: K_FAN_LEAF_STAGE_ID,
            scenario_id: None,
            label: None,
        });
        transitions.push(Transition {
            source_id: 0,
            target_id: fan_id,
            probability: (i as f64) / total_weight,
            annual_discount_rate_override: None,
        });
        transitions.push(Transition {
            source_id: fan_id,
            target_id: leaf_id,
            probability: 1.0,
            annual_discount_rate_override: None,
        });
    }
    // A reversed declaration order must resolve to the identical canonical node
    // graph (build_node_graph sorts nodes by id, out-edges by target) — the input
    // for the enumerated engine's declaration-order-invariance gate.
    if reversed {
        nodes.reverse();
        transitions.reverse();
    }
    PolicyGraph {
        graph_type: PolicyGraphType::FiniteHorizon,
        annual_discount_rate: 0.0,
        transitions,
        nodes,
        stage_discount_rate_overrides: HashMap::new(),
        season_map: None,
    }
}

/// One study stage at `(index, id)` with a single 744h block, storage state
/// enabled (the hydro's storage is a genuine cut-bearing state across the K-fan),
/// `branching_factor: 1` — every scale/routing signal in the K-fan comes from the
/// declared node branching, never from within-node opening variance.
///
/// # Panics
///
/// Never in practice — see the rationale below.
#[allow(clippy::expect_used)]
// Rationale: the literal calendar dates below are valid by construction
// (checked at write time); `from_ymd_opt` only returns `None` for an
// out-of-range calendar date.
fn k_fan_stage(index: usize, id: i32) -> Stage {
    Stage {
        index,
        id,
        start_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
        end_date: NaiveDate::from_ymd_opt(2024, 2, 1).expect("valid date"),
        season_id: None,
        blocks: vec![Block {
            index: 0,
            name: "S".to_string(),
            duration_hours: 744.0,
        }],
        block_mode: BlockMode::Parallel,
        state_config: StageStateConfig {
            storage: true,
            inflow_lags: false,
        },
        risk_config: StageRiskConfig::Expectation,
        scenario_config: ScenarioSourceConfig {
            branching_factor: 1,
            noise_method: NoiseMethod::Saa,
        },
    }
}

/// A single-hydro, single-bus [`System`] over the 3-stage K-fan calendar: hydro
/// storage/inflow/turbine dynamics against a bus deficit fallback, so every
/// visited node solves a genuine (non-degenerate) LP with real dual activity.
///
/// # Panics
///
/// Never in practice — see the rationale below.
#[allow(clippy::too_many_lines, clippy::expect_used)]
// Rationale: one linear entity/bounds/penalties assembly, mirroring
// `deterministic.rs`'s `build_system` — splitting further would scatter the
// literal the caller reads as a whole. The calendar dates are valid by
// construction, and `SystemBuilder::build` only errors on a malformed system,
// which every literal below avoids.
fn k_fan_system(k: usize, reversed: bool) -> System {
    fan_or_chain_system(3, k_fan_policy_graph(k, reversed))
}

/// Shared single-hydro/single-bus study over `n_stages` stages (each a 744h
/// block, `branching_factor: 1`), driven by an explicit `policy_graph` — the
/// declared K-fan (`n_stages == 3`) or, with an empty graph, the chain
/// degeneracy the enumerated engine's single-path (count-1) 2-rank stub needs.
#[allow(clippy::too_many_lines, clippy::expect_used)]
fn fan_or_chain_system(n_stages: usize, policy_graph: PolicyGraph) -> System {
    let bus_id = EntityId(1);
    let hydro_id = EntityId(2);

    let bus = Bus {
        id: bus_id,
        name: "B".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
        deficit_segments: vec![DeficitSegment {
            depth_mw: None,
            cost_per_mwh: 500.0,
        }],
        excess_cost: 0.0,
    };

    let mut hydro = Hydro {
        unit_groups: Vec::new(),
        id: hydro_id,
        name: "H".to_string(),
        operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).expect("valid date"),
        downstream_id: None,
        travel_time_hours: None,
        entry_stage_id: None,
        exit_stage_id: None,
        min_storage_hm3: 0.0,
        max_storage_hm3: 200.0,
        min_outflow_m3s: 0.0,
        max_outflow_m3s: None,
        generation_model: HydroGenerationModel::ConstantProductivity,
        min_turbined_m3s: 0.0,
        max_turbined_m3s: 100.0,
        specific_productivity_mw_per_m3s_per_m: Some(0.5),
        min_generation_mw: 0.0,
        max_generation_mw: 250.0,
        tailrace: None,
        hydraulic_losses: None,
        efficiency: None,
        evaporation_coefficients_mm: None,
        evaporation_reference_volumes_hm3: None,
        diversion: None,
        filling: None,
        penalties: HydroPenalties {
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
        },
    };
    hydro.declare_mirror_unit_group(bus_id);

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let stages: Vec<_> = (0..n_stages).map(|i| k_fan_stage(i, i as i32)).collect();

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let inflow_models: Vec<_> = (0..n_stages)
        .map(|i| InflowModel {
            hydro_id,
            stage_id: i as i32,
            mean_m3s: 60.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        })
        .collect();

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let load_models: Vec<_> = (0..n_stages)
        .map(|i| LoadModel {
            bus_id,
            stage_id: i as i32,
            mean_mw: 80.0,
            std_mw: 0.0,
        })
        .collect();

    let bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
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
                max_storage_hm3: 200.0,
                filling_min_rate_m3s: 0.0,
                water_withdrawal_m3s: 0.0,
            },
            hydro_block: HydroBlockBounds {
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                max_diversion_m3s: None,
            },
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
            },
            line_block: LineBlockBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping_block: PumpingBlockBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract_block: ContractBlockBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );

    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages,
        },
        &PenaltiesDefaults {
            hydro: HydroStagePenalties {
                spillage_cost: 0.01,
                diversion_cost: 0.0,
                turbined_cost: 0.0,
                storage_violation_below_cost: 500.0,
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
            },
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let initial_conditions = InitialConditions {
        storage: vec![HydroStorage {
            hydro_id,
            value_hm3: 100.0,
        }],
        filling_storage: vec![],
        past_anticipated_commitments: vec![],
        recent_observations: vec![],
        past_defluences: vec![],
    };

    SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(stages)
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(bounds)
        .penalties(penalties)
        .initial_conditions(initial_conditions)
        .policy_graph(policy_graph)
        .build()
        .expect("fan_or_chain_system: valid study")
}

/// The `sampled`-mode training [`Config`] for [`k_fan_setup`]: `forward_passes`
/// trajectories per iteration, `max_iterations` fixed via an iteration-limit
/// stopping rule, no `enumerated` selection anywhere.
fn k_fan_config(forward_passes: u32, max_iterations: u32) -> Config {
    Config {
        schema: None,
        state_space: StateSpaceConfig::default(),
        modeling: ModelingConfig {
            inflow_non_negativity: InflowNonNegativityConfig {
                method: InflowNonNegativityMethod::None,
            },
            cost_scale_factor: None,
        },
        training: TrainingConfig {
            enabled: true,
            tree_seed: Some(K_FAN_TREE_SEED),
            stopping_rules: Some(vec![StoppingRuleConfig::IterationLimit {
                limit: max_iterations,
            }]),
            stopping_mode: StoppingMode::Any,
            cut_selection: RowSelectionConfig::default(),
            solver: TrainingSolverConfig::default(),
            parallelism: ParallelismConfig::default(),
            scenario_source: None,
            selection: Some(TrainingSelection::Sampled { forward_passes }),
        },
        upper_bound_evaluation: UpperBoundEvaluationConfig::default(),
        policy: PolicyConfig::default(),
        simulation: IoSimulationConfig::default(),
        exports: ExportsConfig::default(),
        estimation: EstimationConfig::default(),
    }
}

/// The DECOMP K-fan [`StudySetup`], bundled with the fixture parameters
/// callers need to derive their own expected values — never a magic literal,
/// always re-derived from these fields or from `setup.node_graph` directly.
#[derive(Debug)]
pub struct KFanFixture {
    /// The built study: a single-hydro, single-bus system over the declared
    /// K-fan graph, trained `sampled` with a fixed seed.
    pub setup: StudySetup,
    /// Fan-out width: `setup.node_graph` has `k` nodes at
    /// [`K_FAN_BRANCH_STAGE_ID`] and `k` leaves at [`K_FAN_LEAF_STAGE_ID`].
    pub k: usize,
    /// `forward_passes` this fixture was configured with (mirrors
    /// `setup`'s own resolved value; exposed so callers never re-guess it).
    pub forward_passes: u32,
    /// `node_graph::enumerated_scenario_count` for this fixture's graph — the
    /// per-path enumeration the sampled scale assertion must stay strictly
    /// below.
    pub enumerated_scenario_count: u64,
}

/// Build the DECOMP K-fan [`StudySetup`]: a root (stage [`K_FAN_ROOT_STAGE_ID`])
/// fanning into `k` distinct nodes (stage [`K_FAN_BRANCH_STAGE_ID`]) under
/// non-uniform transition weights, each with its own deterministic leaf (stage
/// [`K_FAN_LEAF_STAGE_ID`]) — `num_nodes (2k+1) > n_pools (k+2)` via leaf
/// sharing, and a genuine `k`-node reverse-topological backward level at
/// [`K_FAN_BRANCH_STAGE_ID`] (`backward_cut_levels`). Configured `sampled`
/// (never `enumerated`) with a fixed seed, `forward_passes` trajectories per
/// iteration, `max_iterations` iterations.
///
/// # Panics
///
/// Never in practice — see the rationale below.
#[allow(clippy::expect_used)]
// Rationale: every literal in `k_fan_system`/`k_fan_config` is a hand-checked,
// internally-consistent fixture; `StudySetup::new` only errors on a malformed
// system or config, neither of which this builder can produce, and
// `enumerated_scenario_count` only errors on a `u64` path-product overflow,
// unreachable at this fixture's scale.
#[must_use]
pub fn k_fan_setup(k: usize, forward_passes: u32, max_iterations: u32) -> KFanFixture {
    k_fan_fixture(k, false, k_fan_config(forward_passes, max_iterations))
}

/// The DECOMP K-fan [`StudySetup`] configured `enumerated`: the forward
/// distribution is the exhaustive all-paths engine, so `forward_passes` is
/// resolved by the node graph to `enumerated_scenario_count` (= `k`), not a
/// caller value. Otherwise identical to [`k_fan_setup`] (fixed seed, iteration
/// limit `max_iterations`).
///
/// # Panics
///
/// Never in practice — as [`k_fan_setup`].
#[must_use]
pub fn k_fan_setup_enumerated(k: usize, max_iterations: u32) -> KFanFixture {
    k_fan_fixture(k, false, k_fan_config_enumerated(max_iterations))
}

/// [`k_fan_setup_enumerated`] with the node/transition declaration order
/// reversed. `build_node_graph`'s canonical sort must recover the identical
/// graph, so an enumerated run over this fixture is bit-for-bit identical to
/// the canonical one — the declaration-order-invariance gate.
#[must_use]
pub fn k_fan_setup_enumerated_reversed(k: usize, max_iterations: u32) -> KFanFixture {
    k_fan_fixture(k, true, k_fan_config_enumerated(max_iterations))
}

/// A single-path (count-1) `enumerated` chain study: a 2-stage,
/// `branching_factor: 1` chain (empty policy graph) whose enumerated path count
/// resolves to `1`. The faithful fixture for the enumerated engine's 2-rank
/// stub — `RankOf2` only faithfully simulates rank 0 of 2 when `forward_passes
/// == 1` (rank 1 then genuinely does nothing), matching the backward by-node
/// gates.
///
/// # Panics
///
/// Never in practice — as [`k_fan_setup`].
#[allow(clippy::expect_used)]
#[must_use]
pub fn single_path_enumerated_setup(max_iterations: u32) -> StudySetup {
    let system = fan_or_chain_system(2, PolicyGraph::default());
    let config = k_fan_config_enumerated(max_iterations);
    let stochastic = build_stochastic_context(
        &system,
        K_FAN_SEED,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("single_path_enumerated_setup: build_stochastic_context must succeed");
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    StudySetup::new(&system, &config, stochastic, hydro_models)
        .expect("single_path_enumerated_setup: StudySetup::new must succeed")
}

/// Shared build for [`k_fan_setup`]/[`k_fan_setup_enumerated`]: builds the
/// stochastic context and study for `config` and derives the fixture's exposed
/// counts from the resolved study (never a caller literal).
// `config` is taken by value so callers pass an owned builder result inline; the
// body only borrows it for `StudySetup::new`.
#[allow(clippy::expect_used, clippy::needless_pass_by_value)]
#[must_use]
fn k_fan_fixture(k: usize, reversed: bool, config: Config) -> KFanFixture {
    let system = k_fan_system(k, reversed);
    let stochastic = build_stochastic_context(
        &system,
        K_FAN_SEED,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("k_fan_fixture: build_stochastic_context must succeed");
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);

    let setup = StudySetup::new(&system, &config, stochastic, hydro_models)
        .expect("k_fan_fixture: StudySetup::new must succeed");
    let enumerated = enumerated_scenario_count(&setup.node_graph)
        .expect("k_fan_fixture: enumerated_scenario_count must not overflow at this scale");

    KFanFixture {
        forward_passes: setup.loop_params.forward_passes,
        setup,
        k,
        enumerated_scenario_count: enumerated,
    }
}

/// The `enumerated`-mode training [`Config`] for [`k_fan_setup_enumerated`]:
/// `selection = enumerated` (no explicit `forward_passes` — the graph derives
/// it), fixed seed, iteration-limit stopping rule.
fn k_fan_config_enumerated(max_iterations: u32) -> Config {
    let mut config = k_fan_config(1, max_iterations);
    config.training.selection = Some(TrainingSelection::Enumerated {});
    config
}

/// Attempt to build the K-fan study with `simulation.selection = enumerated`,
/// returning the setup result unpanicked — the enumerated `K >= 2` simulation
/// rejection is a `Result`, not a fixture, so callers assert on the `Err`.
///
/// # Errors
///
/// Returns whatever `StudySetup::new` returns for the enumerated-simulation
/// config (a [`SddpError::Validation`] on the `K >= 2` graph).
///
/// # Panics
///
/// Never in practice: `build_stochastic_context` is infallible for this
/// hand-checked fixture (only the `StudySetup::new` result is returned).
#[allow(clippy::expect_used)]
pub fn try_k_fan_simulation_enumerated(k: usize) -> Result<StudySetup, SddpError> {
    let system = k_fan_system(k, false);
    let mut config = k_fan_config(1, 1);
    config.simulation.enabled = true;
    config.simulation.selection = Some(SimulationSelection::Enumerated {});
    let stochastic = build_stochastic_context(
        &system,
        K_FAN_SEED,
        None,
        &[],
        &[],
        OpeningTreeInputs::default(),
        ClassSchemes {
            inflow: Some(SamplingScheme::InSample),
            load: Some(SamplingScheme::InSample),
            ncs: Some(SamplingScheme::InSample),
        },
    )
    .expect("try_k_fan_simulation_enumerated: build_stochastic_context must succeed");
    let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
    StudySetup::new(&system, &config, stochastic, hydro_models)
}
