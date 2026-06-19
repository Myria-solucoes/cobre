use std::collections::HashMap;

use cobre_core::{EntityId, Stage, System};
use cobre_solver::StageTemplate;
use cobre_stochastic::normal::precompute::PrecomputedNormal;
use cobre_stochastic::par::precompute::PrecomputedPar;

use crate::error::SddpError;
use crate::hydro_models::{EvaporationModelSet, ProductionModelSet, ResolvedProductionModel};
use crate::inflow_method::InflowNonNegativityMethod;
use crate::resolved_parameters::ResolvedParameters;
use crate::setup::template_postprocess::{
    compute_cumulative_discount_factors, compute_per_stage_discount_factors,
};

use super::layout::{ResolvedTables, StageLayout, TemplateBuildCtx};
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
    /// Per-stage pumping-flow column start indices.
    ///
    /// `pumping_col_starts[stage_idx]` is the column index of the first
    /// pumping-flow variable for that stage, sourced from
    /// `StageLayout::col_pumping_start` — never from `StageIndexer::pumping_flow`,
    /// which is a permanent `0..0` sentinel. Pumping columns are block-major:
    /// `pumping_col_starts[stage_idx] + p_idx * n_blks + blk`.
    pub pumping_col_starts: Vec<usize>,
    /// Per-stage pumping-station counts.
    ///
    /// `n_pumping_per_stage[stage_idx]` is the number of pumping stations
    /// contributing columns at that stage, sourced from `StageLayout::n_pumping`.
    pub n_pumping_per_stage: Vec<usize>,
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
    ///
    /// Private with a getter/setter pair: [`build_stage_templates`] leaves this
    /// a `1.0`-placeholder; the real factors are written by
    /// [`StageTemplates::set_discount_factors`] in the postprocess step. A `pub`
    /// field would let an external caller read the placeholder as if it were the
    /// discounted value — silently yielding undiscounted NPV. Read via
    /// [`StageTemplates::discount_factors`].
    discount_factors: Vec<f64>,
    /// Cumulative discount factor at each stage for reporting.
    ///
    /// `cumulative_discount_factors[0] = 1.0` (present value).
    /// `cumulative_discount_factors[t] = cumulative_discount_factors[t-1] * discount_factors[t-1]`
    /// for `t >= 1`.
    ///
    /// Length equals `templates.len()` (one entry per study stage). The
    /// anticipated-decision predicate is strict (`stage_idx + K_i < n_stages`),
    /// so every active delivery stage satisfies `delivery_stage in [0, n_stages)`.
    /// The present value of stage `t`'s immediate cost is
    /// `cumulative_discount_factors[t] * immediate_cost_t`.
    ///
    /// Private for the same reason as [`StageTemplates::discount_factors`]:
    /// derived from it by [`StageTemplates::set_discount_factors`], so it shares
    /// the placeholder window. Read via
    /// [`StageTemplates::cumulative_discount_factors`].
    cumulative_discount_factors: Vec<f64>,
}

impl StageTemplates {
    /// All-empty [`StageTemplates`] for a study with zero stages.
    ///
    /// Every per-stage collection is empty; only `n_hydros` (the stride into
    /// `noise_scale`) carries through, since it is a system-level count that is
    /// well-defined even when no stage templates are built. Used by the
    /// empty-study early return in [`build_stage_templates`].
    #[must_use]
    pub(crate) fn empty(n_hydros: usize) -> Self {
        Self {
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
            pumping_col_starts: Vec::new(),
            n_pumping_per_stage: Vec::new(),
            active_ncs_indices: Vec::new(),
            diversion_upstream: HashMap::new(),
            hydro_productivities_per_stage: Vec::new(),
            discount_factors: Vec::new(),
            cumulative_discount_factors: Vec::new(),
        }
    }

    /// Per-stage one-step discount factors (read access).
    ///
    /// See the [`discount_factors`](StageTemplates#structfield.discount_factors)
    /// field: the slice is a `1.0`-placeholder until
    /// [`StageTemplates::set_discount_factors`] runs in the postprocess step.
    #[must_use]
    pub(crate) fn discount_factors(&self) -> &[f64] {
        &self.discount_factors
    }

    /// Cumulative discount factors for reporting (read access).
    ///
    /// See the
    /// [`cumulative_discount_factors`](StageTemplates#structfield.cumulative_discount_factors)
    /// field: the slice is a `1.0`-placeholder until
    /// [`StageTemplates::set_discount_factors`] runs in the postprocess step.
    #[must_use]
    pub(crate) fn cumulative_discount_factors(&self) -> &[f64] {
        &self.cumulative_discount_factors
    }

    /// Install the real per-stage discount factors and derive the cumulative
    /// factors from them in one call.
    ///
    /// The cumulative vector is always recomputed here from `per_stage`
    /// (`D_0 = 1.0`, `D_t = D_{t-1} * d_{t-1}`), so the two slices cannot drift
    /// out of step: a caller cannot set per-stage factors while leaving a stale
    /// cumulative vector behind. Called once by the postprocess step
    /// (`setup::template_postprocess`).
    pub(crate) fn set_discount_factors(&mut self, per_stage: Vec<f64>) {
        self.cumulative_discount_factors = compute_cumulative_discount_factors(&per_stage);
        self.discount_factors = per_stage;
    }
}

/// Per-stage outputs of [`build_single_stage_template`].
///
/// One field per datum the per-stage build emits; produced by
/// [`build_single_stage_template`] and consumed by
/// [`assemble_stage_templates_output`], which transposes a
/// `Vec<StageBuildOutput>` into the parallel per-stage `Vec`s of
/// [`StageTemplates`]. Adding a per-stage datum is one field here plus one
/// transpose line in the assembler — not a new tuple element threaded through
/// the loop and a parallel argument.
pub(super) struct StageBuildOutput {
    /// Structural LP template for the stage.
    pub template: StageTemplate,
    /// Row index of the first water-balance constraint (the `PatchBuffer`
    /// noise-injection `base_row`).
    pub stage_base_row: usize,
    /// Row index of the first load-balance constraint (load-noise patches).
    pub load_balance_row_start: usize,
    /// Active generic-constraint row metadata for the stage.
    pub gc_entries: Vec<GenericConstraintRowEntry>,
    /// Column index of the first NCS generation variable.
    pub ncs_col_start: usize,
    /// Number of active NCS entities at the stage.
    pub ncs_count: usize,
    /// System-level indices of the active NCS entities, in entity-ID order.
    pub ncs_active: Vec<usize>,
    /// Column index of the first pumping-flow variable (sourced from
    /// `StageLayout::col_pumping_start`).
    pub pumping_col_start: usize,
    /// Number of pumping stations contributing columns at the stage (sourced
    /// from [`StageLayout::n_pumping`]).
    pub n_pumping: usize,
}

/// Construct a [`StageTemplate`] for a single study stage.
///
/// Returns a [`StageBuildOutput`] bundling the template, the two base-row
/// offsets (water-balance for `PatchBuffer` noise injection, load-balance for
/// load-noise patches), the generic constraint row entries, NCS metadata
/// (column start, count, and active system indices), and pumping metadata
/// (column start and station count).
pub(super) fn build_single_stage_template(
    ctx: &TemplateBuildCtx<'_>,
    stage: &Stage,
    stage_idx: usize,
) -> StageBuildOutput {
    let layout = StageLayout::new(ctx, stage, stage_idx);
    let stage_base_row = layout.row_water_balance_start();
    let load_balance_row_start = layout.row_load_balance_start();

    let (col_lower, mut col_upper, mut objective) =
        matrix::fill_stage_columns(ctx, stage, stage_idx, &layout);
    let (mut row_lower, mut row_upper) = matrix::fill_stage_rows(ctx, stage, stage_idx, &layout);
    let mut col_entries = matrix::build_stage_matrix_entries(ctx, stage, stage_idx, &layout);

    // Fill generic constraint rows, slack columns, and CSC entries.
    {
        let mut buffers = matrix::LpMatrixBuffers {
            col_entries: &mut col_entries,
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
    // Use `layout.col_theta()` so the correct index is read from the augmented
    // indexer even when `n_anticipated > 0` shifts theta past the anticipated
    // state block.
    let theta_col = layout.col_theta();
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

    let template = StageTemplate {
        num_cols: layout.num_cols,
        num_rows: layout.num_rows,
        num_nz: col_entries.iter().map(Vec::len).sum(),
        col_starts,
        row_indices,
        values,
        col_lower,
        col_upper,
        objective,
        row_lower,
        row_upper,
        n_state: layout.n_state(),
        n_transfer,
        n_dual_relevant: layout.n_dual_relevant,
        n_hydro: layout.n_h,
        max_par_order: layout.lag_order,
        col_scale: Vec::new(),
        row_scale: Vec::new(),
    };

    StageBuildOutput {
        template,
        stage_base_row,
        load_balance_row_start,
        gc_entries: layout.generic_constraint_rows,
        ncs_col_start: layout.col_ncs_start,
        ncs_count: layout.n_ncs,
        ncs_active: layout.active_ncs_indices,
        pumping_col_start: layout.col_pumping_start,
        n_pumping: layout.n_pumping,
    }
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
/// three stage-level columns are added per hydro (evaporation outflow,
/// `f_evap_plus`, `f_evap_minus`).  The evaporation-outflow column is bounded
/// symmetrically `[-q_max, +q_max]` so a negative value can absorb net rainfall
/// input on the lake surface; `f_evap_plus` and `f_evap_minus` are bounded
/// `[0, +inf)`.  The evaporation-outflow column carries objective coefficient
/// 0.0; the violation slacks carry the evaporation penalty.  One equality
/// constraint row is added per evaporation hydro with
/// `row_lower == row_upper == intercept_m3s`.
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
/// let par_lp = PrecomputedPar::build(&[], &[], &[], None).expect("empty ok");
/// let normal_lp = cobre_stochastic::normal::precompute::PrecomputedNormal::default();
/// let hydro_models = PrepareHydroModelsResult::default_from_system(&system);
/// let resolved_parameters = ResolvedParameters::default();
/// // No stages → empty result.
/// let result = build_stage_templates(&system, method, &par_lp, &normal_lp,
///                                    &hydro_models.production, &hydro_models.evaporation,
///                                    &resolved_parameters)
///     .expect("empty system ok");
/// assert!(result.templates.is_empty());
/// ```
pub fn build_stage_templates(
    system: &System,
    inflow_method: InflowNonNegativityMethod,
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
        return Ok(StageTemplates::empty(n_hydros));
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
    let mut stage_outputs = Vec::with_capacity(n_study);
    for (stage_idx, stage) in study_stages.iter().enumerate() {
        stage_outputs.push(build_single_stage_template(&ctx, stage, stage_idx));
    }

    Ok(assemble_stage_templates_output(
        stage_outputs,
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
    inflow_method: InflowNonNegativityMethod,
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

    // Pumping stations are ID-sorted at `System` build time
    // (`SystemBuilder::build` sorts every entity Vec by `id.0`), so
    // `System::pumping_stations` returns them in canonical order. Iterate that
    // slice in slot order — NOT declaration order — to build `pumping_pos`,
    // mirroring `hydro_pos`/`bus_pos`; this upholds the declaration-order
    // bit-determinism rule.
    let pumping_stations = system.pumping_stations();
    let pumping_pos: HashMap<EntityId, usize> = pumping_stations
        .iter()
        .enumerate()
        .map(|(i, p)| (p.id, i))
        .collect();
    let n_pumping = pumping_stations.len();
    // The resolved-bounds table sizes the `pumping` Vec from the same station
    // count; a divergence means the bounds resolution and the entity slice
    // disagree on how many stations exist (a resolution bug), not a benign
    // empty-system case. Fail fast rather than silently reserving the wrong
    // number of pumping-flow columns.
    debug_assert_eq!(
        n_pumping,
        system.bounds().n_pumping(),
        "pumping_stations.len() ({}) != bounds.n_pumping() ({}): resolved-bounds \
         station count disagrees with the entity slice",
        n_pumping,
        system.bounds().n_pumping()
    );

    let load_bus_indices = collect_load_bus_indices(system, &bus_pos);

    let max_par_order: usize = system
        .inflow_models()
        .iter()
        .filter(|m| m.stage_id >= 0)
        .map(|m| m.ar_coefficients.len())
        .max()
        .unwrap_or(0)
        .max(par_lp.max_order());

    // Compute anticipated-thermal metadata in declaration order.
    // For each thermal with `anticipated_config.is_some()`, record its global
    // index and per-plant lead_stages (K_i).
    let mut anticipated_thermal_indices: Vec<usize> = Vec::new();
    let mut anticipated_lead_stages: Vec<usize> = Vec::new();
    for (t_idx, thermal) in system.thermals().iter().enumerate() {
        if let Some(cfg) = thermal.anticipated_config.as_ref() {
            anticipated_thermal_indices.push(t_idx);
            // u32 always fits in usize on supported 32-bit and 64-bit targets.
            anticipated_lead_stages.push(cfg.lead_stages as usize);
        }
    }
    let n_anticipated = anticipated_thermal_indices.len();
    let k_max = anticipated_lead_stages.iter().copied().max().unwrap_or(0);

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

    // Pre-compute discount factors and total stage hours before the per-stage
    // template loop so that `fill_anticipated_decision_objective` can read them
    // from the ctx at LP build time (before postprocess_templates runs).
    //
    // Both arrays have length `n_study_stages` exactly. The anticipated-decision
    // predicate is strict (`stage_idx + K_i < n_stages`), so every active
    // delivery stage satisfies `delivery_stage in [0, n_stages)` — no phantom
    // boundary entry is needed.
    let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();
    let per_stage_discount =
        compute_per_stage_discount_factors(&study_stages, system.policy_graph());
    let cumulative_discount_factors = compute_cumulative_discount_factors(&per_stage_discount);
    let total_hours_per_stage: Vec<f64> = study_stages
        .iter()
        .map(|s| s.blocks.iter().map(|b| b.duration_hours).sum())
        .collect();

    debug_assert_eq!(
        cumulative_discount_factors.len(),
        study_stages.len(),
        "cumulative_discount_factors length must equal n_study_stages"
    );
    debug_assert_eq!(
        total_hours_per_stage.len(),
        study_stages.len(),
        "total_hours_per_stage length must equal n_study_stages"
    );

    let ctx = TemplateBuildCtx {
        hydros,
        thermals: system.thermals(),
        lines: system.lines(),
        buses,
        load_models: system.load_models(),
        cascade: system.cascade(),
        resolved: ResolvedTables {
            bounds: system.bounds(),
            penalties: system.penalties(),
            resolved_generic_bounds: system.resolved_generic_bounds(),
            resolved_load_factors: system.resolved_load_factors(),
            resolved_exchange_factors: system.resolved_exchange_factors(),
            resolved_ncs_bounds: system.resolved_ncs_bounds(),
            resolved_ncs_factors: system.resolved_ncs_factors(),
            resolved_parameters,
        },
        hydro_pos,
        thermal_pos,
        line_pos,
        bus_pos,
        par_lp,
        production_models,
        evaporation_models,
        generic_constraints: system.generic_constraints(),
        non_controllable_sources: system.non_controllable_sources(),
        pumping_stations,
        pumping_pos,
        n_pumping,
        diversion_upstream,
        n_hydros,
        n_thermals: system.thermals().len(),
        n_lines: system.lines().len(),
        n_buses: buses.len(),
        max_par_order,
        n_anticipated,
        k_max,
        anticipated_lead_stages,
        anticipated_thermal_indices,
        has_penalty: n_hydros > 0 && inflow_method.has_slack_columns(),
        cumulative_discount_factors,
        total_hours_per_stage,
    };

    (ctx, load_bus_indices, diversion_upstream_output)
}

/// Assemble the final [`StageTemplates`] from the per-stage build outputs.
///
/// Transposes the `Vec<StageBuildOutput>` (one entry per study stage) into the
/// parallel per-stage `Vec`s of [`StageTemplates`] in a single pass, then
/// computes the noise-scale, zeta, block-hour, hydro-productivity, and discount
/// arrays and packages everything into the `StageTemplates` returned by
/// [`build_stage_templates`].
///
/// Called once, immediately after the per-stage loop completes.
// Rationale: the remaining args are genuinely separate inputs — the per-stage
// `Vec<StageBuildOutput>` plus the build context (`study_stages`, `ctx`, `par_lp`),
// the cross-stage scalar dims, and the two whole-run outputs (`load_bus_indices`,
// `diversion_upstream_output`). They have distinct lifetimes and ownership (some
// borrowed, some owned), so bundling them into one struct would buy nothing on this
// single-call cold path while obscuring the transpose inputs.
#[allow(clippy::too_many_arguments)]
fn assemble_stage_templates_output(
    stage_outputs: Vec<StageBuildOutput>,
    load_bus_indices: Vec<usize>,
    diversion_upstream_output: HashMap<EntityId, Vec<usize>>,
    study_stages: &[&cobre_core::Stage],
    ctx: &TemplateBuildCtx<'_>,
    par_lp: &PrecomputedPar,
    n_hydros: usize,
    n_load_buses: usize,
    n_study: usize,
) -> StageTemplates {
    // Transpose the per-stage outputs into the parallel Vecs in one pass,
    // preserving the per-stage push order: index `s` of every parallel Vec
    // refers to the same stage, which the assembled StageTemplates relies on.
    let mut templates = Vec::with_capacity(n_study);
    let mut base_rows = Vec::with_capacity(n_study);
    let mut load_balance_row_starts = Vec::with_capacity(n_study);
    let mut generic_constraint_row_entries = Vec::with_capacity(n_study);
    let mut ncs_col_starts = Vec::with_capacity(n_study);
    let mut n_ncs_per_stage = Vec::with_capacity(n_study);
    let mut pumping_col_starts = Vec::with_capacity(n_study);
    let mut n_pumping_per_stage = Vec::with_capacity(n_study);
    let mut active_ncs_indices_per_stage = Vec::with_capacity(n_study);
    for out in stage_outputs {
        templates.push(out.template);
        base_rows.push(out.stage_base_row);
        load_balance_row_starts.push(out.load_balance_row_start);
        generic_constraint_row_entries.push(out.gc_entries);
        ncs_col_starts.push(out.ncs_col_start);
        n_ncs_per_stage.push(out.ncs_count);
        pumping_col_starts.push(out.pumping_col_start);
        n_pumping_per_stage.push(out.n_pumping);
        active_ncs_indices_per_stage.push(out.ncs_active);
    }

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
        pumping_col_starts,
        n_pumping_per_stage,
        active_ncs_indices: active_ncs_indices_per_stage,
        diversion_upstream: diversion_upstream_output,
        hydro_productivities_per_stage,
        // Discount factors are 1.0-placeholders here; the real per-stage and
        // cumulative factors are installed by StageTemplates::set_discount_factors
        // in the postprocess step (setup::template_postprocess). Lengths match
        // n_study: the strict anticipated-decision predicate
        // (`stage_idx + K_i < n_stages`) guarantees every delivery lookup falls
        // within `[0, n_stages)`.
        discount_factors: vec![1.0; n_study],
        cumulative_discount_factors: vec![1.0; n_study],
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    clippy::doc_overindented_list_items
)]
mod tests {
    use chrono::NaiveDate;
    use cobre_core::{
        AnticipatedConfig, Block, BlockMode, BoundsCountsSpec, BoundsDefaults, Bus,
        BusStagePenalties, ContractStageBounds, DeficitSegment, EntityId, Hydro,
        HydroGenerationModel, HydroPenalties, HydroStageBounds, HydroStagePenalties,
        LineStageBounds, LineStagePenalties, LoadModel, NcsStagePenalties, NoiseMethod,
        PenaltiesCountsSpec, PenaltiesDefaults, PumpingStageBounds, PumpingStation, ResolvedBounds,
        ResolvedPenalties, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
        SystemBuilder, Thermal, ThermalStageBounds,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    use crate::hydro_models::PrepareHydroModelsResult;
    use crate::inflow_method::InflowNonNegativityMethod;
    use crate::resolved_parameters::ResolvedParameters;

    // ── Fixtures ─────────────────────────────────────────────────────────────

    fn default_hydro_bounds() -> HydroStageBounds {
        HydroStageBounds {
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            min_generation_mw: 0.0,
            max_generation_mw: 250.0,
            max_diversion_m3s: None,
            filling_inflow_m3s: 0.0,
            water_withdrawal_m3s: 0.0,
        }
    }

    fn default_hydro_penalties() -> HydroStagePenalties {
        HydroStagePenalties {
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
        }
    }

    /// Build a one-bus system with exactly the thermals provided.
    ///
    /// Uses one study stage with a single block of 744 hours and no hydros.
    fn system_with_thermals(thermals: Vec<Thermal>) -> cobre_core::System {
        let n_thermals = thermals.len();
        let n_stages = 1_usize;

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let stages: Vec<Stage> = vec![Stage {
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
        }];

        let load_models = vec![LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 0.0,
        }];

        // k_max for the anticipated thermals in this system
        let k_max = thermals
            .iter()
            .filter_map(|t| t.anticipated_config.as_ref())
            .map(|c| c.lead_stages as usize)
            .max()
            .unwrap_or(0);

        let resolved_bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
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
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(thermals)
            .stages(stages)
            .load_models(load_models)
            .bounds(resolved_bounds)
            .penalties(penalties)
            .build()
            .expect("system_with_thermals: valid system")
    }

    /// Build empty [`ResolvedParameters`] (no parameters).
    fn empty_resolved_params() -> ResolvedParameters {
        ResolvedParameters {
            per_param: vec![],
            id_to_slot: vec![],
        }
    }

    /// All-zero per-plant [`HydroPenalties`] for fixture hydros.
    fn hydro_penalties_zero() -> HydroPenalties {
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

    /// Minimal independent (no-downstream) hydro for pumping-station refs.
    fn fixture_hydro(id: i32) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: hydro_penalties_zero(),
        }
    }

    /// Build a one-bus, two-hydro system with the supplied pumping stations.
    ///
    /// `SystemBuilder::build` sorts every entity Vec by `id.0`, so passing
    /// stations out of declaration order exercises the canonical-ordering
    /// guarantee that `build_template_build_ctx` relies on when threading the
    /// slice into `ctx.pumping_stations`/`pumping_pos`. The two hydros and bus
    /// exist solely to satisfy pumping-station reference validation.
    fn system_with_pumping_stations(stations: Vec<PumpingStation>) -> cobre_core::System {
        let n_pumping = stations.len();
        let n_hydros = 2_usize;
        let n_stages = 1_usize;

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let hydros = vec![fixture_hydro(1), fixture_hydro(2)];

        let stages: Vec<Stage> = vec![Stage {
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
        }];

        let load_models = vec![LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw: 100.0,
            std_mw: 0.0,
        }];

        let resolved_bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros,
                n_thermals: 0,
                n_lines: 0,
                n_pumping,
                n_contracts: 0,
                n_stages,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                    cost_per_mwh: 0.0,
                },
                line: LineStageBounds {
                    direct_mw: 0.0,
                    reverse_mw: 0.0,
                },
                pumping: PumpingStageBounds {
                    min_flow_m3s: 0.0,
                    max_flow_m3s: 100.0,
                },
                contract: ContractStageBounds {
                    min_mw: 0.0,
                    max_mw: 0.0,
                    price_per_mwh: 0.0,
                },
            },
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .hydros(hydros)
            .pumping_stations(stations)
            .stages(stages)
            .load_models(load_models)
            .bounds(resolved_bounds)
            .penalties(penalties)
            .build()
            .expect("system_with_pumping_stations: valid system")
    }

    /// Build a pumping station with the given id (bus/hydro refs fixed to the
    /// fixture entities; flow window and consumption are non-degenerate).
    fn fixture_pumping_station(id: i32) -> PumpingStation {
        PumpingStation {
            id: EntityId(id),
            name: format!("P{id}"),
            bus_id: EntityId(1),
            source_hydro_id: EntityId(1),
            destination_hydro_id: EntityId(2),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s: 0.0,
            max_flow_m3s: 100.0,
        }
    }

    // ── Pumping data threaded into TemplateBuildCtx ────────────────────────────

    /// Stations declared out of ID order are exposed ID-sorted on the ctx, and
    /// `pumping_pos` maps each station id to its slot in that sorted slice.
    ///
    /// Declaration order `[30, 10, 20]` must become `[10, 20, 30]` on the ctx
    /// (the canonical sort applied by `SystemBuilder::build`), with
    /// `pumping_pos = {10->0, 20->1, 30->2}`.
    #[test]
    fn build_template_build_ctx_pumping_stations_id_sorted_and_pos_mapped() {
        let stations = vec![
            fixture_pumping_station(30),
            fixture_pumping_station(10),
            fixture_pumping_station(20),
        ];
        let system = system_with_pumping_stations(stations);
        let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
        let par_lp = PrecomputedPar::default();
        let resolved_params = empty_resolved_params();

        let (ctx, _, _) = super::build_template_build_ctx(
            &system,
            InflowNonNegativityMethod::None,
            &par_lp,
            &hydro_result.production,
            &hydro_result.evaporation,
            &resolved_params,
        );

        let ids: Vec<i32> = ctx.pumping_stations.iter().map(|p| p.id.0).collect();
        assert_eq!(
            ids,
            vec![10, 20, 30],
            "ctx.pumping_stations must be ID-sorted regardless of declaration order"
        );

        assert_eq!(
            ctx.pumping_pos.len(),
            3,
            "pumping_pos has one entry per station"
        );
        assert_eq!(ctx.pumping_pos[&EntityId(10)], 0);
        assert_eq!(ctx.pumping_pos[&EntityId(20)], 1);
        assert_eq!(ctx.pumping_pos[&EntityId(30)], 2);

        // The position map must agree with the slot order of the sorted slice.
        for (slot, station) in ctx.pumping_stations.iter().enumerate() {
            assert_eq!(
                ctx.pumping_pos[&station.id], slot,
                "pumping_pos[{:?}] must equal its slot in the sorted slice",
                station.id
            );
        }
    }

    /// `ctx.n_pumping` equals `pumping_stations.len()` and the resolved-bounds
    /// station count, and that count is the source `StageLayout` reserves from.
    #[test]
    fn build_template_build_ctx_n_pumping_matches_slice_and_bounds() {
        let stations = vec![fixture_pumping_station(7), fixture_pumping_station(3)];
        let system = system_with_pumping_stations(stations);
        let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
        let par_lp = PrecomputedPar::default();
        let resolved_params = empty_resolved_params();

        let (ctx, _, _) = super::build_template_build_ctx(
            &system,
            InflowNonNegativityMethod::None,
            &par_lp,
            &hydro_result.production,
            &hydro_result.evaporation,
            &resolved_params,
        );

        assert_eq!(
            ctx.n_pumping,
            ctx.pumping_stations.len(),
            "n_pumping == slice len"
        );
        assert_eq!(ctx.n_pumping, 2, "two stations were declared");
        assert_eq!(
            ctx.n_pumping,
            ctx.resolved.bounds.n_pumping(),
            "ctx.n_pumping must agree with the resolved-bounds station count"
        );

        // The re-pointed ctx count flows through to the layout: StageLayout reads
        // its `n_pumping` from `EquipmentCounts.n_pumping = ctx.n_pumping`. (The
        // block-major column reservation itself is pinned by the layout-module
        // test `pumping_layout_reserves_block_major_columns`.)
        let stage = system
            .stages()
            .iter()
            .find(|s| s.id >= 0)
            .expect("one study stage");
        let layout = super::super::layout::StageLayout::new(&ctx, stage, 0);
        assert_eq!(
            layout.n_pumping, ctx.n_pumping,
            "StageLayout.n_pumping must equal the ctx-sourced count"
        );
    }

    /// `build_stage_templates` records the layout-owned pumping column base for
    /// every stage: `pumping_col_starts[t]` equals
    /// `StageLayout::new(..).col_pumping_start`, and `n_pumping_per_stage[t]`
    /// equals `StageLayout::new(..).n_pumping`.
    ///
    /// This pins the threading contract the simulation extraction pipeline reads
    /// from: the column base is sourced from the layout, never from
    /// `StageIndexer::pumping_flow` (a permanent `0..0` sentinel).
    #[test]
    fn build_stage_templates_records_layout_pumping_col_start_per_stage() {
        let stations = vec![fixture_pumping_station(5), fixture_pumping_station(2)];
        let system = system_with_pumping_stations(stations);
        let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
        let par_lp = PrecomputedPar::default();
        let normal_lp = cobre_stochastic::normal::precompute::PrecomputedNormal::default();
        let resolved_params = empty_resolved_params();

        let templates = super::build_stage_templates(
            &system,
            InflowNonNegativityMethod::None,
            &par_lp,
            &normal_lp,
            &hydro_result.production,
            &hydro_result.evaporation,
            &resolved_params,
        )
        .expect("build_stage_templates: valid system");

        // Rebuild the ctx once so each stage's StageLayout can be reconstructed
        // and compared against the recorded per-stage value.
        let (ctx, _, _) = super::build_template_build_ctx(
            &system,
            InflowNonNegativityMethod::None,
            &par_lp,
            &hydro_result.production,
            &hydro_result.evaporation,
            &resolved_params,
        );
        let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();

        assert_eq!(templates.pumping_col_starts.len(), study_stages.len());
        assert_eq!(templates.n_pumping_per_stage.len(), study_stages.len());
        for (t, stage) in study_stages.iter().enumerate() {
            let layout = super::super::layout::StageLayout::new(&ctx, stage, t);
            assert_eq!(
                templates.pumping_col_starts[t], layout.col_pumping_start,
                "stage {t}: pumping_col_starts must equal layout.col_pumping_start"
            );
            assert_eq!(
                templates.n_pumping_per_stage[t], layout.n_pumping,
                "stage {t}: n_pumping_per_stage must equal layout.n_pumping"
            );
            assert_eq!(
                templates.n_pumping_per_stage[t], 2,
                "stage {t}: two stations were declared"
            );
        }
    }

    // ── AC-1 ─────────────────────────────────────────────────────────────────

    /// AC-1: `build_template_build_ctx` populates anticipated metadata for a
    /// system with `T_a`(K=2), `T_b`(no anticipated), `T_c`(K=3).
    ///
    /// Expected: `n_anticipated`=2, `k_max`=3, `anticipated_lead_stages`=[2,3],
    /// `anticipated_thermal_indices`=[0,2].
    #[test]
    fn build_template_build_ctx_populates_anticipated_metadata() {
        let thermals = vec![
            Thermal {
                id: EntityId(1),
                name: "T_a".to_string(),
                bus_id: EntityId(1),
                entry_stage_id: None,
                exit_stage_id: None,
                cost_per_mwh: 10.0,
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                anticipated_config: Some(AnticipatedConfig { lead_stages: 2 }),
            },
            Thermal {
                id: EntityId(2),
                name: "T_b".to_string(),
                bus_id: EntityId(1),
                entry_stage_id: None,
                exit_stage_id: None,
                cost_per_mwh: 20.0,
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                anticipated_config: None,
            },
            Thermal {
                id: EntityId(3),
                name: "T_c".to_string(),
                bus_id: EntityId(1),
                entry_stage_id: None,
                exit_stage_id: None,
                cost_per_mwh: 30.0,
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                anticipated_config: Some(AnticipatedConfig { lead_stages: 3 }),
            },
        ];
        let system = system_with_thermals(thermals);
        let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
        let par_lp = PrecomputedPar::default();
        let resolved_params = empty_resolved_params();

        let (ctx, _, _) = super::build_template_build_ctx(
            &system,
            InflowNonNegativityMethod::None,
            &par_lp,
            &hydro_result.production,
            &hydro_result.evaporation,
            &resolved_params,
        );

        assert_eq!(ctx.n_anticipated, 2, "n_anticipated");
        assert_eq!(ctx.k_max, 3, "k_max");
        assert_eq!(
            ctx.anticipated_lead_stages,
            vec![2, 3],
            "anticipated_lead_stages"
        );
        assert_eq!(
            ctx.anticipated_thermal_indices,
            vec![0, 2],
            "anticipated_thermal_indices"
        );
    }

    // ── AC-2 ─────────────────────────────────────────────────────────────────

    /// AC-2: `build_template_build_ctx` returns zeroed metadata when no
    /// thermal has `anticipated_config`.
    #[test]
    fn build_template_build_ctx_zero_anticipated_when_none() {
        let thermals = vec![
            Thermal {
                id: EntityId(1),
                name: "T1".to_string(),
                bus_id: EntityId(1),
                entry_stage_id: None,
                exit_stage_id: None,
                cost_per_mwh: 10.0,
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                anticipated_config: None,
            },
            Thermal {
                id: EntityId(2),
                name: "T2".to_string(),
                bus_id: EntityId(1),
                entry_stage_id: None,
                exit_stage_id: None,
                cost_per_mwh: 20.0,
                min_generation_mw: 0.0,
                max_generation_mw: 100.0,
                anticipated_config: None,
            },
        ];
        let system = system_with_thermals(thermals);
        let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
        let par_lp = PrecomputedPar::default();
        let resolved_params = empty_resolved_params();

        let (ctx, _, _) = super::build_template_build_ctx(
            &system,
            InflowNonNegativityMethod::None,
            &par_lp,
            &hydro_result.production,
            &hydro_result.evaporation,
            &resolved_params,
        );

        assert_eq!(ctx.n_anticipated, 0, "n_anticipated");
        assert_eq!(ctx.k_max, 0, "k_max");
        assert!(
            ctx.anticipated_lead_stages.is_empty(),
            "anticipated_lead_stages"
        );
        assert!(
            ctx.anticipated_thermal_indices.is_empty(),
            "anticipated_thermal_indices"
        );
    }

    // ── Real declaration-order-invariance probe ──

    /// Build a 5-stage 3-thermal system used by the order-invariance probe.
    ///
    /// Three thermals (canonical EntityId order, since `SystemBuilder::build`
    /// sorts by `EntityId`):
    /// - `id=1`: anticipated K=2, max=120 MW, cost=50 $/MWh
    /// - `id=2`: anticipated K=3, max=80 MW, cost=40 $/MWh
    /// - `id=3`: standard thermal (no anticipation), max=200 MW, cost=500 $/MWh
    ///
    /// `ResolvedBounds` is populated with per-thermal stage costs/limits matching
    /// the per-thermal declarations (the default `BoundsDefaults::thermal` is uniform,
    /// so a probe that relied on defaults would be trivial — distinct per-thermal
    /// stage data is required to expose any latent order-dependence in the LP fill).
    ///
    /// `n_stages = 5` ensures both anticipated decisions are active at `stage_idx=0`
    /// (strict gate `t + K_i < n_stages` -> `2 < 5` and `3 < 5`).
    fn anticipated_invariance_system() -> cobre_core::System {
        let thermals = vec![
            Thermal {
                id: EntityId(1),
                name: "T_ant_k2".to_string(),
                bus_id: EntityId(1),
                entry_stage_id: None,
                exit_stage_id: None,
                cost_per_mwh: 50.0,
                min_generation_mw: 0.0,
                max_generation_mw: 120.0,
                anticipated_config: Some(AnticipatedConfig { lead_stages: 2 }),
            },
            Thermal {
                id: EntityId(2),
                name: "T_ant_k3".to_string(),
                bus_id: EntityId(1),
                entry_stage_id: None,
                exit_stage_id: None,
                cost_per_mwh: 40.0,
                min_generation_mw: 0.0,
                max_generation_mw: 80.0,
                anticipated_config: Some(AnticipatedConfig { lead_stages: 3 }),
            },
            Thermal {
                id: EntityId(3),
                name: "T_backup".to_string(),
                bus_id: EntityId(1),
                entry_stage_id: None,
                exit_stage_id: None,
                cost_per_mwh: 500.0,
                min_generation_mw: 0.0,
                max_generation_mw: 200.0,
                anticipated_config: None,
            },
        ];

        let n_thermals = thermals.len();
        let n_stages = 5_usize;
        let k_max = 3_usize;

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
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
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|s| LoadModel {
                bus_id: EntityId(1),
                stage_id: s as i32,
                mean_mw: 150.0,
                std_mw: 0.0,
            })
            .collect();

        let mut resolved_bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
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
        );

        // Per-thermal stage bounds (distinct so a permutation actually changes
        // the LP coefficients). The bounds table is indexed [thermal_idx][stage_idx]
        // with a stage axis of length `n_stages + k_max` (delivery-stage padding).
        let stage_axis_len = resolved_bounds.thermal_stage_axis_len();
        for t_idx in 0..n_thermals {
            for s_idx in 0..stage_axis_len {
                let tb = resolved_bounds.thermal_bounds_mut(t_idx, s_idx);
                match t_idx {
                    0 => {
                        tb.max_generation_mw = 120.0;
                        tb.cost_per_mwh = 50.0;
                    }
                    1 => {
                        tb.max_generation_mw = 80.0;
                        tb.cost_per_mwh = 40.0;
                    }
                    2 => {
                        tb.max_generation_mw = 200.0;
                        tb.cost_per_mwh = 500.0;
                    }
                    _ => unreachable!("only 3 thermals"),
                }
            }
        }

        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(thermals)
            .stages(stages)
            .load_models(load_models)
            .bounds(resolved_bounds)
            .penalties(penalties)
            .build()
            .expect("anticipated_invariance_system: valid system")
    }

    /// Compare two `StageTemplate`s for bit-for-bit equivalence after applying
    /// the swap-(0,1) permutation on anticipated-decision columns,
    /// anticipated-state ring-buffer columns/rows (slot-major), and
    /// anticipated-fishing rows.
    ///
    /// `dec_start_a` / `dec_start_b`: column index of `col_anticipated_decision_start`
    /// in each template.
    /// `state_start_a` / `state_start_b`: column index of
    /// `col_anticipated_state_start` in each template.
    /// `n_ant`: number of anticipated plants (must be 2 for this swap).
    /// `k_max`: ring-buffer slots per plant.
    /// `fish_start_a` / `fish_start_b`: row index of `row_anticipated_fishing_start`.
    /// `n_fish_rows`: number of active fishing rows at this stage (0..=n_ant).
    ///
    /// Strategy: build the column-permutation `col_perm` such that
    /// `tpl_a.column[col_perm[j]]` corresponds to `tpl_b.column[j]`, and the row
    /// permutation `row_perm` analogously. Then assert that the permuted dense
    /// LP (bounds, objective, full coefficient matrix) matches `tpl_b` bitwise.
    ///
    /// Uses dense matrix expansion for clarity; the templates are tiny
    /// (`num_cols ~ 20-50`, `num_rows ~ 10-20`) so the O(n^2) memory cost is fine.
    #[allow(clippy::too_many_arguments)]
    fn assert_lp_equivalence_after_anticipated_swap(
        tpl_a: &cobre_solver::StageTemplate,
        tpl_b: &cobre_solver::StageTemplate,
        dec_start_a: usize,
        dec_start_b: usize,
        state_start_a: usize,
        state_start_b: usize,
        state_out_start_a: usize,
        state_out_start_b: usize,
        n_ant: usize,
        k_max: usize,
        fish_start_a: usize,
        fish_start_b: usize,
        n_fish_rows: usize,
        def_row_start_a: usize,
        def_row_start_b: usize,
        n_def_rows: usize,
        stage_idx: usize,
    ) {
        assert_eq!(
            tpl_a.num_cols, tpl_b.num_cols,
            "stage {stage_idx}: num_cols"
        );
        assert_eq!(
            tpl_a.num_rows, tpl_b.num_rows,
            "stage {stage_idx}: num_rows"
        );
        assert_eq!(tpl_a.num_nz, tpl_b.num_nz, "stage {stage_idx}: num_nz");
        assert_eq!(n_ant, 2, "this helper requires n_ant == 2");

        // Build column permutation: `col_perm[j] = i` means tpl_a column `i`
        // corresponds to tpl_b column `j`. Identity outside the anticipated regions.
        let mut col_perm: Vec<usize> = (0..tpl_a.num_cols).collect();
        // Swap the two anticipated_decision columns.
        col_perm[dec_start_b] = dec_start_a + 1;
        col_perm[dec_start_b + 1] = dec_start_a;
        // Swap the two anticipated_state_out columns (one per plant, plant-indexed).
        col_perm[state_out_start_b] = state_out_start_a + 1;
        col_perm[state_out_start_b + 1] = state_out_start_a;
        // Swap anticipated_state columns at each ring-buffer slot. Slot-major
        // layout: column for slot `s`, plant `p` = state_start + s * n_ant + p.
        for s in 0..k_max {
            col_perm[state_start_b + s * n_ant] = state_start_a + s * n_ant + 1;
            col_perm[state_start_b + s * n_ant + 1] = state_start_a + s * n_ant;
        }

        // Build row permutation: identity outside anticipated_fishing and
        // anticipated_state_out_def rows. State fixing now uses column bounds
        // (no row equalities), so there are no state-fixing rows to permute.
        let mut row_perm: Vec<usize> = (0..tpl_a.num_rows).collect();
        if n_fish_rows == 2 {
            row_perm[fish_start_b] = fish_start_a + 1;
            row_perm[fish_start_b + 1] = fish_start_a;
        }
        // Under the always-active fishing predicate every anticipated plant
        // emits exactly one fishing row at every stage; this branch handles the
        // historical case of a partial active set still encountered in legacy
        // fixtures that pre-date the predicate flip. The SAME plant is active
        // in both LPs — but at LOCAL index 0 in one and
        // LOCAL index 1 in the other. The fishing-row index differs but corresponds
        // to the same plant's constraint. The mapping is still a single-row swap
        // when applicable.
        if n_fish_rows == 1 {
            // The single active fishing row in tpl_a corresponds to the single
            // active fishing row in tpl_b (same plant, different local index).
            row_perm[fish_start_b] = fish_start_a;
        }
        // Anticipated-state-out definition rows: one per active plant (strict gate).
        // When both plants are active, swap rows 0 and 1 (plant order changes).
        // When only one plant is active, the single def row maps identity-wise
        // (the active plant appears at local index 0 in both ctx_a and ctx_b).
        if n_def_rows == 2 {
            row_perm[def_row_start_b] = def_row_start_a + 1;
            row_perm[def_row_start_b + 1] = def_row_start_a;
        }
        if n_def_rows == 1 {
            row_perm[def_row_start_b] = def_row_start_a;
        }

        // Dense bound/objective comparison: tpl_a[col_perm[j]] == tpl_b[j].
        for j in 0..tpl_a.num_cols {
            let a = col_perm[j];
            assert_eq!(
                tpl_a.col_lower[a].to_bits(),
                tpl_b.col_lower[j].to_bits(),
                "stage {stage_idx}: col_lower mismatch at permuted col {j} <- {a}"
            );
            assert_eq!(
                tpl_a.col_upper[a].to_bits(),
                tpl_b.col_upper[j].to_bits(),
                "stage {stage_idx}: col_upper mismatch at permuted col {j} <- {a}"
            );
            assert_eq!(
                tpl_a.objective[a].to_bits(),
                tpl_b.objective[j].to_bits(),
                "stage {stage_idx}: objective mismatch at permuted col {j} <- {a}"
            );
        }
        for i in 0..tpl_a.num_rows {
            let ra = row_perm[i];
            assert_eq!(
                tpl_a.row_lower[ra].to_bits(),
                tpl_b.row_lower[i].to_bits(),
                "stage {stage_idx}: row_lower mismatch at permuted row {i} <- {ra}"
            );
            assert_eq!(
                tpl_a.row_upper[ra].to_bits(),
                tpl_b.row_upper[i].to_bits(),
                "stage {stage_idx}: row_upper mismatch at permuted row {i} <- {ra}"
            );
        }

        // Dense matrix comparison: expand CSC to dense, apply permutation,
        // assert bit-equality. Tiny LPs (~50x20) so the O(n^2) cost is fine.
        let dense_a = csc_to_dense(tpl_a);
        let dense_b = csc_to_dense(tpl_b);
        for i in 0..tpl_a.num_rows {
            for j in 0..tpl_a.num_cols {
                let va = dense_a[row_perm[i]][col_perm[j]];
                let vb = dense_b[i][j];
                assert_eq!(
                    va.to_bits(),
                    vb.to_bits(),
                    "stage {stage_idx}: coefficient mismatch at row {i} col {j} \
                     (permuted from row {} col {} in tpl_a)",
                    row_perm[i],
                    col_perm[j],
                );
            }
        }
    }

    /// Expand a CSC `StageTemplate` to a dense `Vec<Vec<f64>>`.
    fn csc_to_dense(tpl: &cobre_solver::StageTemplate) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0_f64; tpl.num_cols]; tpl.num_rows];
        for j in 0..tpl.num_cols {
            let start = tpl.col_starts[j] as usize;
            let end = tpl.col_starts[j + 1] as usize;
            for k in start..end {
                let row = tpl.row_indices[k] as usize;
                dense[row][j] = tpl.values[k];
            }
        }
        dense
    }

    /// **Invariance probe** — direct LP-construction layer test.
    ///
    /// Verifies that the LP templates produced by [`build_single_stage_template`]
    /// are equivalent under a permutation of the `anticipated_thermal_indices` /
    /// `anticipated_lead_stages` arrays.
    ///
    /// ## Why this test exists
    ///
    /// The integration test at
    /// `crates/cobre-sddp/tests/declaration_order_invariance_anticipated.rs`
    /// is a tautology: it builds two `System`s with thermals declared in
    /// different orders, but `SystemBuilder::build()` sorts by `EntityId` so
    /// both systems present identical canonical input downstream. That test
    /// proves **determinism** (same canonical input -> same output), not
    /// **invariance** (different declaration orders -> same canonical result).
    ///
    /// The Cobre hard rule on declaration-order invariance requires bit-for-bit
    /// identical results regardless of input entity ordering. This unit test
    /// targets the **actual** code path that the canonical sort masks: it
    /// directly constructs a `TemplateBuildCtx` with a permuted (yet internally
    /// consistent) pair of `(anticipated_thermal_indices, anticipated_lead_stages)`
    /// arrays and confirms that the resulting LP coefficients are equivalent
    /// (modulo the expected swap of the anticipated-decision columns,
    /// anticipated-state ring-buffer columns/rows, and anticipated-fishing rows).
    ///
    /// ## Method
    ///
    /// 1. Build a system with two anticipated thermals (K=2 and K=3) plus one
    ///    standard backup thermal, with **distinct** per-thermal stage costs
    ///    and bounds (uniform defaults would trivially pass).
    /// 2. Call `build_template_build_ctx` to obtain `ctx_a` with the canonical
    ///    ordering `anticipated_thermal_indices = [0, 1]`,
    ///    `anticipated_lead_stages = [2, 3]`.
    /// 3. Manually construct `ctx_b` by swapping both arrays in lockstep:
    ///    `anticipated_thermal_indices = [1, 0]`,
    ///    `anticipated_lead_stages = [3, 2]`.
    /// 4. Build single-stage templates for both contexts at stages 0, 2, and 3.
    ///    Under the always-active fishing predicate every anticipated plant
    ///    emits one fishing row at every stage, so all sampled stages carry
    ///    the same number of fishing rows; the anticipated-decision active set
    ///    is independent of the fishing predicate and still depends on
    ///    `t + K_i < T` at each stage.
    /// 5. Assert LP equivalence under the canonical swap permutation
    ///    (column swap on anticipated_decision and slot-major state, row swap
    ///    on state-fixing and fishing rows when both plants are present).
    #[test]
    fn lp_template_invariant_under_anticipated_index_permutation() {
        let system = anticipated_invariance_system();
        // Canonical sort places thermals as [id=1, id=2, id=3].
        assert_eq!(system.thermals().len(), 3);
        assert_eq!(system.thermals()[0].id.0, 1);
        assert_eq!(system.thermals()[1].id.0, 2);
        assert_eq!(system.thermals()[2].id.0, 3);

        let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
        let par_lp = PrecomputedPar::default();
        let resolved_params = ResolvedParameters {
            per_param: vec![],
            id_to_slot: vec![],
        };

        let (ctx_a, _, _) = super::build_template_build_ctx(
            &system,
            InflowNonNegativityMethod::None,
            &par_lp,
            &hydro_result.production,
            &hydro_result.evaporation,
            &resolved_params,
        );

        // Sanity: ctx_a uses canonical ordering.
        assert_eq!(ctx_a.n_anticipated, 2);
        assert_eq!(ctx_a.k_max, 3);
        assert_eq!(ctx_a.anticipated_thermal_indices, vec![0, 1]);
        assert_eq!(ctx_a.anticipated_lead_stages, vec![2, 3]);

        // Construct ctx_b: a clone of ctx_a with the two anticipated arrays
        // swapped in lockstep. Both arrays must be permuted by the same
        // permutation to preserve the (thermal_idx, K_i) pairing.
        let ctx_b = super::super::layout::TemplateBuildCtx {
            hydros: ctx_a.hydros,
            thermals: ctx_a.thermals,
            lines: ctx_a.lines,
            buses: ctx_a.buses,
            load_models: ctx_a.load_models,
            cascade: ctx_a.cascade,
            resolved: super::super::layout::ResolvedTables {
                bounds: ctx_a.resolved.bounds,
                penalties: ctx_a.resolved.penalties,
                resolved_generic_bounds: ctx_a.resolved.resolved_generic_bounds,
                resolved_load_factors: ctx_a.resolved.resolved_load_factors,
                resolved_exchange_factors: ctx_a.resolved.resolved_exchange_factors,
                resolved_ncs_bounds: ctx_a.resolved.resolved_ncs_bounds,
                resolved_ncs_factors: ctx_a.resolved.resolved_ncs_factors,
                resolved_parameters: ctx_a.resolved.resolved_parameters,
            },
            hydro_pos: ctx_a.hydro_pos.clone(),
            thermal_pos: ctx_a.thermal_pos.clone(),
            line_pos: ctx_a.line_pos.clone(),
            bus_pos: ctx_a.bus_pos.clone(),
            par_lp: ctx_a.par_lp,
            production_models: ctx_a.production_models,
            evaporation_models: ctx_a.evaporation_models,
            generic_constraints: ctx_a.generic_constraints,
            non_controllable_sources: ctx_a.non_controllable_sources,
            pumping_stations: ctx_a.pumping_stations,
            pumping_pos: ctx_a.pumping_pos.clone(),
            n_pumping: ctx_a.n_pumping,
            diversion_upstream: ctx_a.diversion_upstream.clone(),
            n_hydros: ctx_a.n_hydros,
            n_thermals: ctx_a.n_thermals,
            n_lines: ctx_a.n_lines,
            n_buses: ctx_a.n_buses,
            max_par_order: ctx_a.max_par_order,
            n_anticipated: ctx_a.n_anticipated,
            k_max: ctx_a.k_max,
            // The swap: lockstep permutation [0,1] -> [1,0] on both arrays.
            anticipated_lead_stages: vec![
                ctx_a.anticipated_lead_stages[1],
                ctx_a.anticipated_lead_stages[0],
            ],
            anticipated_thermal_indices: vec![
                ctx_a.anticipated_thermal_indices[1],
                ctx_a.anticipated_thermal_indices[0],
            ],
            has_penalty: ctx_a.has_penalty,
            cumulative_discount_factors: ctx_a.cumulative_discount_factors.clone(),
            total_hours_per_stage: ctx_a.total_hours_per_stage.clone(),
        };

        // Sanity: ctx_b really has the swapped ordering.
        assert_eq!(ctx_b.anticipated_thermal_indices, vec![1, 0]);
        assert_eq!(ctx_b.anticipated_lead_stages, vec![3, 2]);

        let study_stages: Vec<_> = system.stages().iter().filter(|s| s.id >= 0).collect();

        // Test multiple stages to cover the active-decision boundary while
        // the always-active fishing predicate keeps the fishing-row count
        // constant at every stage (one row per anticipated plant).
        for stage_idx in [0_usize, 2, 3] {
            let stage = study_stages[stage_idx];

            let tpl_a = super::build_single_stage_template(&ctx_a, stage, stage_idx).template;
            let tpl_b = super::build_single_stage_template(&ctx_b, stage, stage_idx).template;

            // Reconstruct the indexer for tpl_a / tpl_b to find the
            // anticipated_decision / anticipated_state / fishing row offsets.
            // Both templates use the same num_cols/num_rows (the layout depends
            // only on n_anticipated and k_max, both unchanged by the swap).
            let layout_a = super::super::layout::StageLayout::new(&ctx_a, stage, stage_idx);
            let layout_b = super::super::layout::StageLayout::new(&ctx_b, stage, stage_idx);

            // Layout offsets must be identical (they depend only on counts,
            // not on the contents of the anticipated arrays).
            assert_eq!(
                layout_a.anticipated.col_anticipated_decision_start,
                layout_b.anticipated.col_anticipated_decision_start,
                "stage {stage_idx}: dec_start"
            );
            assert_eq!(
                layout_a.col_anticipated_state_start(),
                layout_b.col_anticipated_state_start(),
                "stage {stage_idx}: state_start"
            );
            assert_eq!(
                layout_a.anticipated.row_anticipated_fishing_start,
                layout_b.anticipated.row_anticipated_fishing_start,
                "stage {stage_idx}: fish_start"
            );
            assert_eq!(
                layout_a.anticipated.n_anticipated_fishing_rows,
                layout_b.anticipated.n_anticipated_fishing_rows,
                "stage {stage_idx}: n_fish_rows"
            );

            assert_lp_equivalence_after_anticipated_swap(
                &tpl_a,
                &tpl_b,
                layout_a.anticipated.col_anticipated_decision_start,
                layout_b.anticipated.col_anticipated_decision_start,
                layout_a.col_anticipated_state_start(),
                layout_b.col_anticipated_state_start(),
                layout_a.anticipated.col_anticipated_state_out_start,
                layout_b.anticipated.col_anticipated_state_out_start,
                ctx_a.n_anticipated,
                ctx_a.k_max,
                layout_a.anticipated.row_anticipated_fishing_start,
                layout_b.anticipated.row_anticipated_fishing_start,
                layout_a.anticipated.n_anticipated_fishing_rows,
                layout_a.anticipated.row_anticipated_state_out_def_start,
                layout_b.anticipated.row_anticipated_state_out_def_start,
                layout_a.anticipated.n_anticipated_state_out_def_rows,
                stage_idx,
            );
        }
    }

    // ── StageTemplates::empty ──────────────────────────────────────────────────

    /// `StageTemplates::empty(n)` yields every per-stage collection empty and
    /// records `n_hydros == n`. This pins the all-empty shape the empty-study
    /// early return relies on.
    #[test]
    fn stage_templates_empty_is_all_empty_with_n_hydros() {
        let n = 7_usize;
        let empty = super::StageTemplates::empty(n);

        assert_eq!(empty.n_hydros, n, "empty(n).n_hydros must equal n");
        assert_eq!(empty.n_load_buses, 0, "n_load_buses must be 0");

        assert!(empty.templates.is_empty(), "templates");
        assert!(empty.base_rows.is_empty(), "base_rows");
        assert!(empty.noise_scale.is_empty(), "noise_scale");
        assert!(empty.zeta_per_stage.is_empty(), "zeta_per_stage");
        assert!(
            empty.block_hours_per_stage.is_empty(),
            "block_hours_per_stage"
        );
        assert!(
            empty.load_balance_row_starts.is_empty(),
            "load_balance_row_starts"
        );
        assert!(empty.load_bus_indices.is_empty(), "load_bus_indices");
        assert!(
            empty.generic_constraint_row_entries.is_empty(),
            "generic_constraint_row_entries"
        );
        assert!(empty.ncs_col_starts.is_empty(), "ncs_col_starts");
        assert!(empty.n_ncs_per_stage.is_empty(), "n_ncs_per_stage");
        assert!(empty.pumping_col_starts.is_empty(), "pumping_col_starts");
        assert!(empty.n_pumping_per_stage.is_empty(), "n_pumping_per_stage");
        assert!(empty.active_ncs_indices.is_empty(), "active_ncs_indices");
        assert!(empty.diversion_upstream.is_empty(), "diversion_upstream");
        assert!(
            empty.hydro_productivities_per_stage.is_empty(),
            "hydro_productivities_per_stage"
        );
        assert!(empty.discount_factors().is_empty(), "discount_factors");
        assert!(
            empty.cumulative_discount_factors().is_empty(),
            "cumulative_discount_factors"
        );
    }

    // ── discount-factor placeholder is replaced by the public path ─────────────

    /// Build a 3-stage thermals-only system carrying a non-zero global annual
    /// discount rate. Empty `transitions` means every stage falls back to the
    /// global rate, so the postprocessed per-stage factors are all < 1.0 and the
    /// cumulative vector compounds below the 1.0 placeholder.
    fn discounted_multi_stage_system() -> cobre_core::System {
        use cobre_core::{PolicyGraph, PolicyGraphType};

        let n_stages = 3_usize;
        let thermals = vec![Thermal {
            id: EntityId(1),
            name: "T1".to_string(),
            bus_id: EntityId(1),
            entry_stage_id: None,
            exit_stage_id: None,
            cost_per_mwh: 10.0,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            anticipated_config: None,
        }];
        let n_thermals = thermals.len();

        let bus = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
        };

        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| Stage {
                index: i,
                id: i as i32,
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
            })
            .collect();

        let load_models: Vec<LoadModel> = (0..n_stages)
            .map(|s| LoadModel {
                bus_id: EntityId(1),
                stage_id: s as i32,
                mean_mw: 100.0,
                std_mw: 0.0,
            })
            .collect();

        let resolved_bounds = ResolvedBounds::new(
            &BoundsCountsSpec {
                n_hydros: 0,
                n_thermals,
                n_lines: 0,
                n_pumping: 0,
                n_contracts: 0,
                n_stages,
                k_max: 0,
            },
            &BoundsDefaults {
                hydro: default_hydro_bounds(),
                thermal: ThermalStageBounds {
                    min_generation_mw: 0.0,
                    max_generation_mw: 100.0,
                    cost_per_mwh: 10.0,
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
        );
        let penalties = ResolvedPenalties::new(
            &PenaltiesCountsSpec {
                n_hydros: 0,
                n_buses: 1,
                n_lines: 0,
                n_ncs: 0,
                n_stages,
            },
            &PenaltiesDefaults {
                hydro: default_hydro_penalties(),
                bus: BusStagePenalties { excess_cost: 0.0 },
                line: LineStagePenalties { exchange_cost: 0.0 },
                ncs: NcsStagePenalties {
                    curtailment_cost: 0.0,
                },
            },
        );

        // Non-zero global rate with no per-transition overrides: every stage
        // discounts at the global rate.
        let policy_graph = PolicyGraph {
            graph_type: PolicyGraphType::FiniteHorizon,
            annual_discount_rate: 0.10,
            transitions: Vec::new(),
            season_map: None,
        };

        SystemBuilder::new()
            .buses(vec![bus])
            .thermals(thermals)
            .stages(stages)
            .load_models(load_models)
            .bounds(resolved_bounds)
            .penalties(penalties)
            .policy_graph(policy_graph)
            .build()
            .expect("discounted_multi_stage_system: valid system")
    }

    /// Any `StageTemplates` produced by the public build + postprocess path has
    /// `cumulative_discount_factors().len() == templates.len()` and is no longer
    /// the all-`1.0` placeholder once a non-zero discount rate is in effect.
    ///
    /// The discount fields are private, so an external caller can only ever observe
    /// the postprocessed (discounted) values, never the placeholder
    /// `build_stage_templates` leaves behind.
    #[test]
    fn postprocessed_stage_templates_carry_discounted_factors() {
        let system = discounted_multi_stage_system();
        let hydro_result = PrepareHydroModelsResult::default_from_system(&system);
        let par_lp = PrecomputedPar::default();
        let normal_lp = cobre_stochastic::normal::precompute::PrecomputedNormal::default();
        let resolved_params = empty_resolved_params();

        let mut templates = super::build_stage_templates(
            &system,
            InflowNonNegativityMethod::None,
            &par_lp,
            &normal_lp,
            &hydro_result.production,
            &hydro_result.evaporation,
            &resolved_params,
        )
        .expect("build_stage_templates: valid system");

        // Run the postprocess step that installs the real discount factors —
        // exactly the public path StudySetup drives.
        let _report =
            crate::setup::template_postprocess::postprocess_templates(&mut templates, &system);

        let cumulative = templates.cumulative_discount_factors();
        assert_eq!(
            cumulative.len(),
            templates.templates.len(),
            "cumulative_discount_factors length must equal templates.len() after postprocess"
        );
        assert_eq!(
            cumulative[0], 1.0,
            "cumulative_discount_factors[0] is the present value (1.0)"
        );
        // With a 10% annual rate the later cumulative factors compound strictly
        // below the 1.0 placeholder, so the postprocessed vector is provably not
        // the placeholder the builder hands back.
        assert!(
            cumulative.iter().any(|&d| d < 1.0),
            "postprocessed cumulative factors must drop below the 1.0 placeholder, got {cumulative:?}"
        );
        assert!(
            cumulative[cumulative.len() - 1] < 1.0,
            "the final cumulative factor must be discounted below 1.0, got {}",
            cumulative[cumulative.len() - 1]
        );
    }
}
