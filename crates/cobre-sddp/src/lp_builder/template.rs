use std::collections::HashMap;

use cobre_core::{EntityId, Stage, System};
use cobre_solver::StageTemplate;
use cobre_stochastic::normal::precompute::PrecomputedNormal;
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::error::SddpError;
use crate::hydro_models::{EvaporationModelSet, ProductionModelSet, ResolvedProductionModel};
use crate::indexer::StageIndexer;
use crate::inflow_method::InflowNonNegativityMethod;
use crate::resolved_parameters::ResolvedParameters;

use super::layout::{StageLayout, TemplateBuildCtx};
use super::{COST_SCALE_FACTOR, GenericConstraintRowEntry, matrix, scaling};

/// Outcome of [`build_stage_templates`]: one [`StageTemplate`] per study stage
/// plus the per-stage `base_rows` offsets needed by `PatchBuffer`.
///
/// `base_rows[s]` is the row index of the first water-balance (AR dynamics)
/// constraint in stage `s`.  It equals `template.n_dual_relevant` for every
/// stage (constant when all stages share the same entity set, which is the
/// case for the minimal viable solver).  It is stored per-stage for forward
/// compatibility with stages that have different active entity sets.
#[derive(Debug, Clone)]
pub struct StageTemplates {
    /// One structural LP template per study stage, in stage order.
    pub templates: Vec<StageTemplate>,
    /// Row index of the first water-balance constraint in each stage's LP.
    ///
    /// Length equals `templates.len()`.  Used by `PatchBuffer::fill_forward_patches`
    /// to locate the noise-injection rows (Category 3 patches).
    pub base_rows: Vec<usize>,
    /// Pre-computed noise scale `ζ_stage * σ_{stage,hydro}` for each (stage, hydro) pair.
    ///
    /// Flat array in stage-major layout: `noise_scale[stage * n_hydros + hydro]`.
    /// Length equals `n_study_stages * n_hydros`.
    ///
    /// Used by the forward pass to transform raw standard-normal noise `η` into
    /// the full noise term `ζ*σ*η` before patching the water-balance RHS.
    /// The complete patch value is `ζ*base + ζ*σ*η`, where `ζ*base` is encoded
    /// in the template's `row_lower`/`row_upper` and `ζ*σ*η` is computed by the
    /// caller at each stage using this pre-computed scale.
    pub noise_scale: Vec<f64>,
    /// Per-stage time-conversion factor `ζ = total_hours * M3S_TO_HM3`.
    ///
    /// Length equals `templates.len()`.  Used by the simulation pipeline to
    /// convert the water-balance RHS (in hm³) back to inflow in m³/s for
    /// output reporting: `inflow_m3s = rhs_hm3 / zeta_per_stage[stage]`.
    pub zeta_per_stage: Vec<f64>,
    /// Per-stage block durations in hours.
    ///
    /// `block_hours_per_stage[stage]` is a `Vec<f64>` of length `n_blocks` for
    /// that stage.  Used by the simulation pipeline to convert load-balance
    /// constraint duals from $/MW to $/`MWh`: `spot_price = dual / block_hours`.
    pub block_hours_per_stage: Vec<Vec<f64>>,
    /// Number of hydro plants (N) used to stride into `noise_scale`.
    pub n_hydros: usize,
    /// Per-stage row index of the first load-balance constraint.
    ///
    /// `load_balance_row_starts[s]` equals `row_water_balance_start + n_hydros`
    /// for stage `s`.  Length equals `templates.len()`.  Used by the forward,
    /// backward, and simulation passes to locate load-balance rows for
    /// stochastic load patching.
    pub load_balance_row_starts: Vec<usize>,
    /// Number of buses with stochastic load noise (i.e. with `std_mw > 0`).
    ///
    /// Equals `normal_lp.n_entities()`.  Tells the forward and backward passes
    /// how many load-noise components to extract from the opening tree noise
    /// vector, which carries load noise in indices `[n_hydros, n_hydros + n_load_buses)`.
    pub n_load_buses: usize,
    /// Position in the `buses` slice for each stochastic load bus.
    ///
    /// Length equals `n_load_buses`.  Bus IDs are sorted by [`cobre_core::EntityId`] for
    /// declaration-order invariance.  The forward and backward passes use
    /// `load_bus_indices[i]` to compute the base row index of bus `i` in the
    /// load-balance region: `row = load_balance_row_start + load_bus_indices[i] * n_blks + blk`.
    pub load_bus_indices: Vec<usize>,
    /// Per-stage metadata for active generic constraint rows.
    ///
    /// `generic_constraint_row_entries[s]` contains one
    /// [`GenericConstraintRowEntry`] per active `(constraint, block)` pair at
    /// stage `s`.  Used by the simulation extraction pipeline to map LP
    /// row/column indices back to constraint identity and block.  Empty for
    /// stages with no active generic constraints.
    pub generic_constraint_row_entries: Vec<Vec<GenericConstraintRowEntry>>,
    /// Per-stage NCS column start indices.
    ///
    /// `ncs_col_starts[stage_idx]` is the column index of the first NCS generation
    /// variable for that stage.
    pub ncs_col_starts: Vec<usize>,
    /// Per-stage active NCS counts.
    ///
    /// `n_ncs_per_stage[stage_idx]` is the number of active NCS entities at that stage.
    pub n_ncs_per_stage: Vec<usize>,
    /// Per-stage active NCS system indices.
    ///
    /// `active_ncs_indices[stage_idx]` lists the system-level NCS indices active at
    /// that stage, in entity-ID order.
    pub active_ncs_indices: Vec<Vec<usize>>,
    /// Mapping from target hydro ID to source hydro indices that divert to it.
    ///
    /// Used by the simulation extraction pipeline to compute `diverted_inflow_m3s`.
    /// Empty when no hydros have diversion.
    pub diversion_upstream: HashMap<EntityId, Vec<usize>>,
    /// Per-stage hydro productivities (MW per m³/s) for simulation extraction.
    ///
    /// `hydro_productivities_per_stage[stage][h]` is the productivity of hydro `h`
    /// at stage `stage`, accounting for per-stage overrides.  FPHA hydros have 0.0.
    pub hydro_productivities_per_stage: Vec<Vec<f64>>,
    /// Per-stage one-step discount factor for the transition departing stage `t`.
    ///
    /// `discount_factors[t] = 1 / (1 + r_t)^(Dt / 365.25)` where `r_t` is the
    /// annual discount rate (global or per-transition override) and `Dt` is the
    /// stage duration in days. When `annual_discount_rate == 0.0` and no
    /// per-transition overrides exist, all entries are `1.0`.
    ///
    /// Length equals `templates.len()`. Computed at setup time and applied to
    /// the theta objective coefficient in the LP template.
    pub discount_factors: Vec<f64>,
    /// Cumulative discount factor at each stage for reporting.
    ///
    /// `cumulative_discount_factors[0] = 1.0` (present value).
    /// `cumulative_discount_factors[t] = cumulative_discount_factors[t-1] * discount_factors[t-1]`
    /// for `t >= 1`.
    ///
    /// Length equals `templates.len()`. The present value of stage `t`'s
    /// immediate cost is `cumulative_discount_factors[t] * immediate_cost_t`.
    pub cumulative_discount_factors: Vec<f64>,
}

/// Construct a [`StageTemplate`] for a single study stage.
///
/// Returns the template, the row index of the water-balance block
/// (used as `base_row` by the `PatchBuffer` noise injection), the
/// row index of the load-balance block (used for load-noise patches),
/// the generic constraint row entries for this stage, NCS metadata
/// (column start, count, and active system indices), and z-inflow
/// metadata (row start, column start).
#[allow(clippy::type_complexity)]
pub(super) fn build_single_stage_template(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
) -> (
    StageTemplate,
    usize,
    usize,
    Vec<GenericConstraintRowEntry>,
    usize,
    usize,
    Vec<usize>,
) {
    let layout = StageLayout::new(ctx, stage, stage_idx);
    let stage_base_row = layout.row_water_balance_start;
    let load_balance_row_start = layout.row_load_balance_start;

    let (mut col_lower, mut col_upper, mut objective) =
        matrix::fill_stage_columns(ctx, stage, stage_idx, &layout);
    let (mut row_lower, mut row_upper) = matrix::fill_stage_rows(ctx, stage, stage_idx, &layout);
    let mut col_entries = matrix::build_stage_matrix_entries(ctx, stage, stage_idx, &layout);

    // Fill generic constraint rows, slack columns, and CSC entries.
    {
        let mut buffers = matrix::LpMatrixBuffers {
            col_entries: &mut col_entries,
            _col_lower: &mut col_lower,
            col_upper: &mut col_upper,
            objective: &mut objective,
            row_lower: &mut row_lower,
            row_upper: &mut row_upper,
        };
        matrix::fill_generic_constraint_entries(ctx, stage, stage_idx, &layout, &mut buffers);
    }

    // Scale all monetary objective coefficients for numerical conditioning.
    // The entire SDDP algorithm operates in scaled cost space; outputs
    // are unscaled at the reporting boundary (forward.rs, lower_bound.rs,
    // simulation/pipeline.rs, simulation/extraction.rs).
    //
    // Theta (the future cost approximation variable) must NOT be divided by
    // COST_SCALE_FACTOR.  The Benders cuts enforce `theta >= intercept_scaled`
    // where `intercept_scaled = Q_successor / K`, so theta holds the SCALED
    // future cost.  The LP objective is `sum(c_i/K * x_i) + 1.0 * theta`, and
    // the total scaled objective = (stage_cost + future_cost) / K.  Multiplying
    // by K at the reporting boundary recovers the original monetary cost.
    //
    // If theta were also divided by K its objective coefficient would become
    // 1/K, making the LP objective `stage_cost/K + (1/K)*theta` which, after
    // multiplication by K, gives `stage_cost + future_cost/K` -- wrong.
    let theta_col = StageIndexer::new(ctx.n_hydros, ctx.max_par_order).theta;
    for (i, coeff) in objective.iter_mut().enumerate() {
        if i != theta_col {
            *coeff /= COST_SCALE_FACTOR;
        }
    }

    // Sort each column's entries by row index (CSC invariant).
    for entries in &mut col_entries {
        entries.sort_unstable_by_key(|&(row, _)| row);
    }

    let (col_starts, row_indices, values) = matrix::assemble_csc(&col_entries);

    let n_transfer = ctx.n_hydros * ctx.max_par_order;
    let total_nz = col_entries.iter().map(Vec::len).sum();

    let gc_row_entries = layout.generic_constraint_rows;

    let ncs_col_start = layout.col_ncs_start;
    let n_ncs = layout.n_ncs;
    let ncs_active = layout.active_ncs_indices;

    let template = StageTemplate {
        num_cols: layout.num_cols,
        num_rows: layout.num_rows,
        num_nz: total_nz,
        col_starts,
        row_indices,
        values,
        col_lower,
        col_upper,
        objective,
        row_lower,
        row_upper,
        n_state: layout.n_state,
        n_transfer,
        n_dual_relevant: layout.n_dual_relevant,
        n_hydro: layout.n_h,
        max_par_order: layout.lag_order,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    (
        template,
        stage_base_row,
        load_balance_row_start,
        gc_row_entries,
        ncs_col_start,
        n_ncs,
        ncs_active,
    )
}

/// Collect the bus-slice positions of stochastic load buses.
///
/// Returns bus-position indices (into the buses slice) for every bus that has
/// `std_mw > 0` in any load model, sorted by `EntityId` for declaration-order
/// invariance.  Buses with duplicate IDs across stages are deduplicated.
fn collect_load_bus_indices(system: &System, bus_pos: &HashMap<EntityId, usize>) -> Vec<usize> {
    // `n_load_buses` must equal `normal_lp.n_entities()` in a consistent
    // system; both are derived from buses with std_mw > 0 in the load models.
    let mut ids: Vec<EntityId> = system
        .load_models()
        .iter()
        .filter(|m| m.std_mw > 0.0)
        .map(|m| m.bus_id)
        .collect();
    ids.sort_unstable_by_key(|id| id.0);
    ids.dedup();
    ids.iter()
        .filter_map(|id| bus_pos.get(id).copied())
        .collect()
}

/// Build one [`StageTemplate`] per study stage from a fully loaded [`System`].
///
/// The templates encode the complete structural LP for each SDDP subproblem
/// in CSC format, ready for bulk-loading via `SolverInterface::load_model`.
/// They are constructed once at solver initialisation and shared read-only
/// across all solver threads.
///
/// ## Column and row layout
///
/// See the module-level documentation for the full LP layout.
/// Key dimensions for a stage with N hydros, T thermals, Lines lines,
/// B buses, K blocks per stage, and F FPHA hydros each with M planes:
///
/// - `num_cols` and `num_rows` are computed by `layout::StageLayout` —
///   see `layout.rs` for the authoritative column and row counts
/// - `n_state  = N*(1+L)`
/// - `n_transfer = N*L`  (storage + all lags except the oldest)
/// - `n_dual_relevant = N*(1+L)`  (`z_inflow` definition, water balance, load balance, FPHA,
///   evaporation, operational violation, and generic constraint rows are all structural and
///   non-dual-relevant; only the state-fixing rows contribute to cut gradients)
///
/// ## PAR order and `max_par_order`
///
/// `max_par_order` is the maximum of (a) the maximum AR coefficient count
/// across all hydro inflow models and (b) `par_lp.max_order()`.  The latter
/// is non-classical only when an annual component is present, in which case
/// the precompute widens the lag stride to 12 and the LP must allocate
/// matching column and row slots.  All hydros use the same uniform lag stride
/// `max_par_order` to enable SIMD-friendly contiguous access.
///
/// ## Objective coefficients
///
/// Costs are expressed in `$/MWh` (thermal, deficit, excess, lines) multiplied
/// by the block duration in hours so they integrate to $/block.  Storage, lag,
/// incoming-storage, theta, turbine, and spillage columns carry zero or small
/// regularization costs drawn from the resolved penalty tables.
///
/// When the penalty method is active, each inflow slack column `sigma_inf_h`
/// carries objective coefficient `penalty_cost * total_stage_hours`.
///
/// FPHA generation columns carry objective coefficient 0.0 by default.
///
/// ## Inflow non-negativity
///
/// When `inflow_method.has_slack_columns()` is `true` (i.e., the `Penalty`
/// variant), `N` slack columns `sigma_inf_h >= 0`
/// are appended at the end of the column layout.  Each slack enters the water
/// balance row for hydro `h` with coefficient `+tau_total * M3S_TO_HM3`,
/// acting as virtual inflow that prevents infeasibility when the PAR(p) noise
/// is sufficiently negative.
///
/// ## FPHA hydros
///
/// For hydros whose resolved production model at a given stage is
/// [`ResolvedProductionModel::Fpha`], generation becomes a free variable
/// `g_{h,k} ∈ [0, max_generation_mw]` bounded by M hyperplane constraints:
///
/// ```text
/// g_{h,k} - gamma_v/2*v - gamma_v/2*v_in - gamma_q*q_{h,k} - gamma_s*s_{h,k} <= gamma_0
/// ```
///
/// The `v_in` contribution propagates through the LP via the matrix coefficient
/// `-gamma_v/2` on the incoming-storage column; when `v_in` is fixed by the
/// storage-fixing equality row its value automatically enters the FPHA constraint
/// right-hand side.  No changes to the backward pass or cut extraction are needed.
///
/// Returns `Ok` with empty templates for a system with zero stages.  All
/// entity counts may be zero (valid for degenerate test systems).
///
/// # Errors
///
/// Returns [`SddpError`] if the PAR precomputation data is inconsistent with
/// the system (e.g., a hydro in `par_lp` is not present in `system`), or if
/// the production model set has incompatible dimensions.
///
/// ## Evaporation hydros
///
/// For hydros whose evaporation model is
/// `EvaporationModel::Linearized`,
/// three stage-level columns are added per hydro (`Q_ev`, `f_evap_plus`,
/// `f_evap_minus`), all bounded `[0, +inf)` with objective coefficient 0.0.
/// One equality constraint row is added per evaporation hydro with
/// `row_lower == row_upper == k_evap0`.
///
/// # Examples
///
/// ```
/// use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
/// use cobre_sddp::InflowNonNegativityMethod;
/// use cobre_sddp::hydro_models::PrepareHydroModelsResult;
/// use cobre_sddp::lp_builder::build_stage_templates;
/// use cobre_sddp::resolved_parameters::ResolvedParameters;
/// use cobre_stochastic::par::precompute::PrecomputedPar;
///
/// let bus = Bus {
///     id: EntityId(1),
///     name: "B1".to_string(),
///     deficit_segments: vec![DeficitSegment { depth_mw: None, cost_per_mwh: 1000.0 }],
///     excess_cost: 0.0,
/// };
/// let system = SystemBuilder::new().buses(vec![bus]).build().expect("valid");
/// let method = InflowNonNegativityMethod::None;
/// let par_lp = PrecomputedPar::build(&[], &[], &[]).expect("empty ok");
/// let normal_lp = cobre_stochastic::normal::precompute::PrecomputedNormal::default();
/// let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
/// let resolved_parameters = ResolvedParameters::default();
/// // No stages → empty result.
/// let result = build_stage_templates(&system, &method, &par_lp, &normal_lp,
///                                    &hydro_models.production, &hydro_models.evaporation,
///                                    &resolved_parameters)
///     .expect("empty system ok");
/// assert!(result.templates.is_empty());
/// ```
pub fn build_stage_templates(
    system: &System,
    inflow_method: &InflowNonNegativityMethod,
    par_lp: &PrecomputedPar,
    normal_lp: &PrecomputedNormal,
    production_models: &ProductionModelSet,
    evaporation_models: &EvaporationModelSet,
    resolved_parameters: &ResolvedParameters,
) -> Result<StageTemplates, SddpError> {
    // Only build templates for study stages (id >= 0), in canonical order.
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let n_hydros = system.hydros().len();

    if study_stages.is_empty() {
        return Ok(StageTemplates {
            templates: Vec::new(),
            base_rows: Vec::new(),
            noise_scale: Vec::new(),
            zeta_per_stage: Vec::new(),
            block_hours_per_stage: Vec::new(),
            n_hydros,
            load_balance_row_starts: Vec::new(),
            n_load_buses: 0,
            load_bus_indices: Vec::new(),
            generic_constraint_row_entries: Vec::new(),
            ncs_col_starts: Vec::new(),
            n_ncs_per_stage: Vec::new(),
            active_ncs_indices: Vec::new(),
            diversion_upstream: HashMap::new(),
            hydro_productivities_per_stage: Vec::new(),
            discount_factors: Vec::new(),
            cumulative_discount_factors: Vec::new(),
        });
    }

    // Consistency gate: a non-empty PrecomputedNormal must have the same
    // entity count as the stochastic load buses derived from the system.
    let (ctx, load_bus_indices, diversion_upstream_output) = build_template_build_ctx(
        system,
        inflow_method,
        par_lp,
        production_models,
        evaporation_models,
        resolved_parameters,
    );
    let n_load_buses = load_bus_indices.len();
    debug_assert!(
        normal_lp.n_entities() == 0 || normal_lp.n_entities() == n_load_buses,
        "PrecomputedNormal has {} entities but system has {} stochastic load buses",
        normal_lp.n_entities(),
        n_load_buses
    );

    let n_study = study_stages.len();
    let mut templates = Vec::with_capacity(n_study);
    let mut base_rows = Vec::with_capacity(n_study);
    let mut load_balance_row_starts = Vec::with_capacity(n_study);
    let mut generic_constraint_row_entries = Vec::with_capacity(n_study);
    let mut ncs_col_starts = Vec::with_capacity(n_study);
    let mut n_ncs_per_stage = Vec::with_capacity(n_study);
    let mut active_ncs_indices_per_stage = Vec::with_capacity(n_study);
    for (stage_idx, stage) in study_stages.iter().enumerate() {
        let (
            template,
            stage_base_row,
            load_balance_row_start,
            gc_entries,
            ncs_col_start,
            ncs_count,
            ncs_active,
        ) = build_single_stage_template(&ctx, stage, stage_idx);
        templates.push(template);
        base_rows.push(stage_base_row);
        load_balance_row_starts.push(load_balance_row_start);
        generic_constraint_row_entries.push(gc_entries);
        ncs_col_starts.push(ncs_col_start);
        n_ncs_per_stage.push(ncs_count);
        active_ncs_indices_per_stage.push(ncs_active);
    }

    Ok(assemble_stage_templates_output(
        templates,
        base_rows,
        load_balance_row_starts,
        generic_constraint_row_entries,
        ncs_col_starts,
        n_ncs_per_stage,
        active_ncs_indices_per_stage,
        load_bus_indices,
        diversion_upstream_output,
        &study_stages,
        &ctx,
        par_lp,
        n_hydros,
        n_load_buses,
        n_study,
    ))
}

/// Build the [`TemplateBuildCtx`] and ancillary data needed by the stage loop.
///
/// Constructs position maps (hydro/thermal/line/bus), the diversion-upstream
/// map, and the `TemplateBuildCtx` that is shared across all per-stage builds.
/// Also returns `load_bus_indices` (the bus-slice positions of stochastic load
/// buses) and `diversion_upstream_output` (the clone of the diversion map
/// preserved for the final `StageTemplates` output field).
///
/// Called once per `build_stage_templates` invocation, after the early-return
/// guard for empty systems.
fn build_template_build_ctx<'a>(
    system: &'a System,
    inflow_method: &InflowNonNegativityMethod,
    par_lp: &'a PrecomputedPar,
    production_models: &'a ProductionModelSet,
    evaporation_models: &'a EvaporationModelSet,
    resolved_parameters: &'a ResolvedParameters,
) -> (
    TemplateBuildCtx<'a>,
    Vec<usize>,
    HashMap<EntityId, Vec<usize>>,
) {
    let hydros = system.hydros();
    let buses = system.buses();
    let n_hydros = hydros.len();

    let hydro_pos: HashMap<EntityId, usize> =
        hydros.iter().enumerate().map(|(i, h)| (h.id, i)).collect();
    let thermal_pos: HashMap<EntityId, usize> = system
        .thermals()
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id, i))
        .collect();
    let line_pos: HashMap<EntityId, usize> = system
        .lines()
        .iter()
        .enumerate()
        .map(|(i, l)| (l.id, i))
        .collect();
    let bus_pos: HashMap<EntityId, usize> =
        buses.iter().enumerate().map(|(i, b)| (b.id, i)).collect();

    let load_bus_indices = collect_load_bus_indices(system, &bus_pos);

    let max_par_order: usize = system
        .inflow_models()
        .iter()
        .filter(|m| m.stage_id >= 0)
        .map(|m| m.ar_coefficients.len())
        .max()
        .unwrap_or(0)
        .max(par_lp.max_order());

    // Precompute diversion upstream map: maps target hydro ID -> list of source
    // hydro indices that divert water to it. O(1) lookup in water balance loop.
    // Cloned so the map is available both for LP construction (ctx) and for the
    // simulation extraction pipeline (StageTemplates output).
    let mut diversion_upstream: HashMap<EntityId, Vec<usize>> = HashMap::new();
    for (h_idx, hydro) in hydros.iter().enumerate() {
        if let Some(ref div) = hydro.diversion {
            diversion_upstream
                .entry(div.downstream_id)
                .or_default()
                .push(h_idx);
        }
    }
    let diversion_upstream_output = diversion_upstream.clone();

    let ctx = TemplateBuildCtx {
        hydros,
        thermals: system.thermals(),
        lines: system.lines(),
        buses,
        load_models: system.load_models(),
        cascade: system.cascade(),
        bounds: system.bounds(),
        penalties: system.penalties(),
        hydro_pos,
        thermal_pos,
        line_pos,
        bus_pos,
        par_lp,
        production_models,
        evaporation_models,
        generic_constraints: system.generic_constraints(),
        resolved_generic_bounds: system.resolved_generic_bounds(),
        resolved_load_factors: system.resolved_load_factors(),
        resolved_exchange_factors: system.resolved_exchange_factors(),
        non_controllable_sources: system.non_controllable_sources(),
        resolved_ncs_bounds: system.resolved_ncs_bounds(),
        resolved_ncs_factors: system.resolved_ncs_factors(),
        resolved_parameters,
        diversion_upstream,
        n_hydros,
        n_thermals: system.thermals().len(),
        n_lines: system.lines().len(),
        n_buses: buses.len(),
        max_par_order,
        has_penalty: n_hydros > 0 && inflow_method.has_slack_columns(),
    };

    (ctx, load_bus_indices, diversion_upstream_output)
}

/// Assemble the final [`StageTemplates`] from per-stage loop outputs.
///
/// Computes noise-scale, zeta, block-hour, hydro-productivity, and discount
/// arrays and packages them alongside the per-stage template vectors into the
/// `StageTemplates` struct returned by `build_stage_templates`.
///
/// Called once, immediately after the per-stage loop completes.
// RATIONALE: 15 args are the heterogeneous per-stage accumulator Vecs produced by the
// per-stage build loop, each of a distinct type (templates, base_rows, ncs_col_starts, etc.).
// They cannot be grouped into a context struct without either re-allocating them after the
// loop or wrapping in Option, both of which add cost on this post-loop cold path.
#[allow(clippy::too_many_arguments)]
fn assemble_stage_templates_output(
    templates: Vec<cobre_solver::StageTemplate>,
    base_rows: Vec<usize>,
    load_balance_row_starts: Vec<usize>,
    generic_constraint_row_entries: Vec<Vec<GenericConstraintRowEntry>>,
    ncs_col_starts: Vec<usize>,
    n_ncs_per_stage: Vec<usize>,
    active_ncs_indices_per_stage: Vec<Vec<usize>>,
    load_bus_indices: Vec<usize>,
    diversion_upstream_output: HashMap<EntityId, Vec<usize>>,
    study_stages: &[&cobre_core::Stage],
    ctx: &TemplateBuildCtx<'_>,
    par_lp: &PrecomputedPar,
    n_hydros: usize,
    n_load_buses: usize,
    n_study: usize,
) -> StageTemplates {
    let (noise_scale, zeta_per_stage, block_hours_per_stage) =
        scaling::compute_noise_scale(study_stages, n_hydros, par_lp);

    // Build per-stage productivity arrays for simulation extraction.
    let hydro_productivities_per_stage: Vec<Vec<f64>> = (0..n_study)
        .map(|s| {
            (0..n_hydros)
                .map(|h| match ctx.production_models.model(h, s) {
                    ResolvedProductionModel::ConstantProductivity { productivity } => *productivity,
                    ResolvedProductionModel::Fpha { .. } => 0.0,
                })
                .collect()
        })
        .collect();

    // Default discount factors to 1.0 (no discounting). The actual
    // per-stage factors are computed from the PolicyGraph in
    // StudySetup::from_broadcast_params and overwrite this field.
    let discount_factors = vec![1.0; templates.len()];

    StageTemplates {
        templates,
        base_rows,
        noise_scale,
        zeta_per_stage,
        block_hours_per_stage,
        n_hydros,
        load_balance_row_starts,
        n_load_buses,
        load_bus_indices,
        generic_constraint_row_entries,
        ncs_col_starts,
        n_ncs_per_stage,
        active_ncs_indices: active_ncs_indices_per_stage,
        diversion_upstream: diversion_upstream_output,
        hydro_productivities_per_stage,
        discount_factors,
        // Cumulative factors default to 1.0; overwritten by setup.rs.
        cumulative_discount_factors: vec![1.0; n_study],
    }
}
