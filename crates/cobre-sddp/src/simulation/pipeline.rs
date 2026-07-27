//! Simulation forward pass for policy evaluation.
//!
//! [`simulate`] evaluates the trained policy on scenarios. Scenarios distributed
//! across ranks via two-level distribution; within-rank parallelism via Rayon.
//! Seed domain separated from training. No per-scenario allocations on hot path.

use std::collections::HashMap;
use std::sync::mpsc::{Sender, SyncSender};

use cobre_comm::Communicator;
use cobre_core::commissioning::commissioning_active;
use cobre_core::temporal::StageLagTransition;
use cobre_core::{EntityId, TrainingEvent};
use cobre_solver::ActiveProfile;
use cobre_solver::{SolverInterface, StageTemplate};
use cobre_stochastic::{ClassSampleRequest, ForwardSampler, SampleRequest};

use crate::energy_conversion::EnergyConversionSet;
use crate::error::SddpError::Infeasible;
use crate::error::SddpError::Solver;
use crate::indexer::StudyDimensions;
use crate::lp_builder::GenericConstraintRowEntry;
use crate::lp_builder::StageGeometry;
use crate::noise::DownstreamAccumState;
use crate::noise::LagAccumState;
use crate::noise::accumulate_and_shift_lag_state;
use crate::stage_solve::StageInputs;
use crate::stage_solve::debug_assert_bucket_copy_gap_intact;
use crate::stage_solve::fill_unscaled;
use crate::stage_solve::fill_unscaled_dual;
use crate::stage_solve::run_stage_solve;
use crate::{
    FutureCostFunction, SddpError,
    context::{StageContext, TrainingContext},
    dcs::{DcsSolveContext, build_initial_resident_set, lazy_solve_preloaded},
    indexer::StateSpace,
    simulation::{
        config::SimulationConfig,
        error::SimulationError,
        extraction::EntityCounts,
        extraction::{
            HydroReverseLookup, SolutionView, StageExtractionSpec, ThermalReverseLookup,
            accumulate_category_costs, extract_stage_result_with_lookups,
        },
        types::{ScenarioCategoryCosts, SimulationScenarioResult, SimulationStageResult},
    },
    solver_stats::SolverStatsDelta,
    training::stage_solve_prep::{
        InflowNoise, LoadNoise, OpeningMode, StageSolvePrep, StageSolvePrepParams, StateSource,
    },
    workspace::{CapturedBasis, SolverWorkspace},
};

/// Offset added to the simulation scenario ID before passing to [`ForwardSampler::sample`].
///
/// Separates the simulation seed domain from the training forward pass domain so
/// the two never draw the same noise stream.
pub(crate) const SIMULATION_SEED_OFFSET: u32 = u32::MAX / 2;

/// Per-worker scenario cost accumulation: `(scenario_id, total_cost, category_costs)`.
pub(crate) type WorkerCosts = Vec<(u32, f64, ScenarioCategoryCosts)>;

/// Per-worker solver statistics: `(scenario_id, opening, delta)`.
///
/// `opening` is always `-1` for simulation (no opening loop); the sentinel maps
/// to a NULL `Int32` in the parquet schema.
pub(crate) type WorkerStats = Vec<(u32, i32, SolverStatsDelta)>;

/// Result of a simulation run, containing per-scenario costs and solver statistics.
#[derive(Debug)]
pub struct SimulationRunResult {
    /// Per-scenario `(scenario_id, total_cost, category_costs)`, sorted by `scenario_id`.
    pub costs: Vec<(u32, f64, ScenarioCategoryCosts)>,
    /// Per-scenario `(scenario_id, opening, delta)`, sorted by `scenario_id`.
    pub solver_stats: Vec<(u32, i32, SolverStatsDelta)>,
}

/// Output-related inputs bundled from the caller for [`simulate`].
///
/// Groups the output-channel, unit-conversion arrays, and optional event sender
/// that would otherwise push the argument count of `simulate` beyond seven.
pub struct SimulationOutputSpec<'a> {
    /// Bounded channel used to stream completed scenario results to the caller.
    pub result_tx: &'a SyncSender<SimulationScenarioResult>,

    /// Per-stage productivity factor (hm³/MWh) for converting LP water-balance
    /// RHS values to volumetric inflow (m³/s) in output records.
    pub zeta_per_stage: &'a [f64],

    /// Per-stage block hours used to compute hourly energy from block dispatch.
    pub block_hours_per_stage: &'a [Vec<f64>],

    /// Entity counts for result extraction (hydros, thermals, lines, etc.).
    pub entity_counts: &'a EntityCounts,

    /// Per-stage active generic-constraint row metadata for extraction.
    pub generic_constraint_row_entries: &'a [Vec<GenericConstraintRowEntry>],

    /// Per-stage column index of the first NCS generation variable.
    pub ncs_col_starts: &'a [usize],

    /// NCS column count — a single scalar, identical at every stage: under the
    /// dense layout a dormant NCS keeps its column.
    pub n_ncs: usize,

    /// Per-stage column index of the first pumping-flow variable. The per-stage
    /// `StageLayout` is the sole owner of this base.
    pub pumping_col_starts: &'a [usize],

    /// Pumping-station column count — a single scalar, identical at every stage:
    /// a commissioning-dormant station keeps its column (pinned to `[0, 0]`).
    pub n_pumping: usize,

    /// Per-stage equipment geometry for extraction, from the per-stage
    /// `StageLayout`. A single global stage-0 geometry carries `n_blks`-striped
    /// bases that misread any stage with a differing block count.
    pub geometry_per_stage: &'a [StageGeometry],

    /// Per-station pumping power-consumption rate \[MW/(m³/s)\], ID-sorted
    /// parallel to `entity_counts.pumping_station_ids` and indexed by the SYSTEM
    /// station index — which, under the dense layout, IS the column-block position.
    pub pumping_consumption_mw_per_m3s: &'a [f64],

    /// Per-stage RESOLVED contract price \[$/`MWh`\]: one inner slice per study
    /// stage, flat with the per-stage stride `n_blks` — index `c * n_blks + blk`,
    /// `c` ID-sorted parallel to `entity_counts.contract_ids`. The resolved,
    /// possibly block-overridden `contract_bounds_at_block(c, t, blk).price_per_mwh`
    /// — never the `col_scale`-scaled LP objective.
    pub contract_prices_per_stage: &'a [Vec<f64>],

    /// Direction per contract, ID-sorted parallel to `entity_counts.contract_ids`
    /// (`true` = import). Stage-invariant.
    pub contract_is_import: &'a [bool],

    /// Per-stage NCS entity IDs, in ID-sorted system order (dense — all NCS).
    pub ncs_entity_ids_per_stage: &'a [Vec<i32>],

    /// Map from target hydro ID to source hydro indices that divert to it; empty
    /// when no hydros have diversion.
    pub diversion_upstream: &'a HashMap<EntityId, Vec<usize>>,

    /// Per-stage per-hydro productivity (per-stage override, or base if none);
    /// `0.0` for FPHA hydros.
    pub hydro_productivities_per_stage: &'a [Vec<f64>],

    /// Pre-computed energy-conversion scalars for every `(hydro, stage)` pair.
    pub energy_conversion: &'a EnergyConversionSet,

    /// Minimum storage volume `V_min` per hydro plant (hm³), ID-sorted.
    pub hydro_min_storage_hm3: &'a [f64],

    /// Optional event sender for streaming progress events to the CLI/UI.
    pub event_sender: Option<Sender<TrainingEvent>>,
}

/// Per-scenario context bundled for `process_scenario_stages`.
///
/// Groups the scenario identifiers, scratch buffers, and noise sampler so the
/// processor does not exceed the clippy `too_many_arguments` budget.
pub(crate) struct ScenarioIds<'a> {
    /// Local scenario ID (0-based index within this rank's assigned slice).
    pub(crate) scenario_id: u32,
    /// Global scenario ID passed to `ForwardSampler::sample`, including
    /// [`SIMULATION_SEED_OFFSET`].
    pub(crate) global_scenario: u32,
    /// Total number of stages in the planning horizon.
    pub(crate) num_stages: usize,
    /// Total simulation scenario count, passed to `SampleRequest::total_scenarios`.
    pub(crate) total_scenarios: u32,
    /// Caller-owned buffer for raw noise output (reused across stages).
    pub(crate) raw_noise_buf: &'a mut [f64],
    /// Caller-owned permutation scratch for LHS generation (reused across stages).
    pub(crate) perm_scratch: &'a mut [usize],
    /// Noise sampler used to draw per-stage stochastic values.
    pub(crate) sampler: &'a ForwardSampler<'a>,
}

/// Rebuild the stage `row_lower` slice into `scratch_buf` in original (unscaled)
/// units. An empty `row_scale` means no prescaling, so template rows are copied
/// without per-element division.
fn build_row_lower_unscaled<'a>(
    template_row_lower: &[f64],
    row_scale: &[f64],
    load_rhs_buf: &[f64],
    scratch_buf: &'a mut Vec<f64>,
    n_load_buses: usize,
    load_balance_row_start: usize,
    n_blks: usize,
    load_bus_indices: &[usize],
) -> &'a [f64] {
    scratch_buf.clear();
    scratch_buf.reserve(template_row_lower.len());

    if row_scale.is_empty() {
        scratch_buf.extend_from_slice(template_row_lower);
    } else {
        for (i, &val) in template_row_lower.iter().enumerate() {
            let scale = if i < row_scale.len() && row_scale[i] != 0.0 {
                row_scale[i]
            } else {
                1.0
            };
            scratch_buf.push(val / scale);
        }
    }

    // load_rhs_buf is already in unscaled MW.
    if n_load_buses > 0 && !load_rhs_buf.is_empty() {
        let mut rhs_idx = 0;
        for &bus_pos in load_bus_indices {
            for blk in 0..n_blks {
                scratch_buf[load_balance_row_start + bus_pos * n_blks + blk] =
                    load_rhs_buf[rhs_idx];
                rhs_idx += 1;
            }
        }
    }

    &scratch_buf[..template_row_lower.len()]
}

/// Stage identifiers bundled for `solve_simulation_stage`.
struct SimStageIds {
    /// Stage index (0-based).
    t: usize,
    /// Stage index as `u32` for result records and error messages.
    stage_id_u32: u32,
    /// Scenario ID for error messages.
    scenario_id: u32,
}

/// Load-path inputs bundled for `solve_simulation_stage`.
///
/// The LP is loaded via `load_model(frozen_template)` — the frozen template
/// already embeds all active cut rows as structural rows. No `add_rows` call
/// is needed.
struct SimStageLoadSpec<'a> {
    /// Frozen template for this stage; always populated after the startup re-freeze.
    frozen_template: &'a StageTemplate,
    /// Warm-start basis captured during training at this stage, if any.
    warm_basis: Option<&'a CapturedBasis>,
}

/// Per-stage batched form of [`SimStageLoadSpec`] consumed by
/// `process_scenario_stages`; indexed by stage via [`Self::stage`].
pub(crate) struct SimScenarioLoadSpec<'a> {
    pub(crate) frozen_templates: &'a [StageTemplate],
    pub(crate) stage_bases: &'a [Option<CapturedBasis>],
}

impl<'a> SimScenarioLoadSpec<'a> {
    #[inline]
    fn stage(&self, t: usize) -> SimStageLoadSpec<'a> {
        SimStageLoadSpec {
            frozen_template: &self.frozen_templates[t],
            warm_basis: self.stage_bases.get(t).and_then(Option::as_ref),
        }
    }
}

/// Pre-built reverse-lookup tables for simulation extraction.
///
/// Built once per worker via [`SimLookups::build`] and passed by reference into
/// every per-stage extraction call to eliminate per-`(scenario, stage)`
/// allocations on the hot path.
///
/// Thermal membership is study-invariant (one table); FPHA/evaporation membership
/// is per-`(hydro, stage)`, so a single global hydro lookup would misclassify any
/// stage whose membership differs from stage 0's.
pub(crate) struct SimLookups {
    /// Reverse-lookup table for anticipated thermal indices (study-invariant).
    pub(crate) thermal: ThermalReverseLookup,
    /// Per-stage hydro FPHA/evaporation lookups, indexed by stage.
    pub(crate) hydro_per_stage: Vec<HydroReverseLookup>,
}

impl SimLookups {
    /// Build the reverse-lookup tables from study dimensions, the per-stage
    /// geometry table, and entity counts.
    pub(crate) fn build(
        study_dims: &StudyDimensions,
        geometry_per_stage: &[StageGeometry],
        n_thermals: usize,
        n_hydros: usize,
    ) -> Self {
        Self {
            thermal: ThermalReverseLookup::build(study_dims, n_thermals),
            hydro_per_stage: HydroReverseLookup::build_per_stage(geometry_per_stage, n_hydros),
        }
    }
}

/// Map a stage-solve [`SddpError`](crate::error::SddpError) to a
/// [`SimulationError`], carrying the scenario/stage ids. Shared by the frozen
/// `run_stage_solve` path and the DCS `lazy_solve_preloaded` path so both report
/// failures identically.
fn map_sim_solver_error(e: SddpError, ids: &SimStageIds) -> SimulationError {
    match e {
        Infeasible {
            stage, scenario, ..
        } => {
            #[allow(clippy::cast_possible_truncation)]
            let scenario_id = scenario as u32;
            #[allow(clippy::cast_possible_truncation)]
            let stage_id = stage as u32;
            SimulationError::LpInfeasible {
                scenario_id,
                stage_id,
                solver_message: "LP infeasible".to_string(),
            }
        }
        Solver(other) => SimulationError::SolverError {
            scenario_id: ids.scenario_id,
            stage_id: ids.stage_id_u32,
            solver_message: other.to_string(),
        },
        other => SimulationError::SolverError {
            scenario_id: ids.scenario_id,
            stage_id: ids.stage_id_u32,
            solver_message: format!("{other}"),
        },
    }
}

/// Solve one stage for one simulation scenario, updating workspace in-place.
///
/// Returns `(immediate_cost, SimulationStageResult)`. The warm-start basis is a
/// read-only, per-stage training artifact, so determinism is preserved. Under
/// dynamic cut-selection the stage is solved lazily against the cut pool from the
/// cut-free base template (the frozen cut rows are unused); the realized primal is
/// identical at the optimum by exactness.
// RATIONALE: the sequential per-stage steps cannot split without fragmenting the
// per-stage invariant tracking.
#[allow(clippy::too_many_lines)]
fn solve_simulation_stage<S: SolverInterface>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    fcf: &FutureCostFunction,
    training_ctx: &TrainingContext<'_>,
    load_spec: &SimStageLoadSpec<'_>,
    output: &SimulationOutputSpec<'_>,
    ids: &SimStageIds,
    lookups: &SimLookups,
    raw_noise: &[f64],
) -> Result<(f64, SimulationStageResult), SimulationError> {
    let TrainingContext {
        state,
        study_dims,
        stochastic,
        dcs,
        ..
    } = training_ctx;
    let dcs = *dcs;
    let t = ids.t;
    // DCS loads the cut-free base so its fresh CutRowMap owns the resident cut
    // subset; loading the frozen template would double-append the embedded cut rows.
    if dcs.is_some() {
        ws.solver.load_model(&ctx.templates[t]);
    } else {
        ws.solver.load_model(load_spec.frozen_template);
    }
    let prep_params = StageSolvePrepParams {
        state_source: StateSource(&ws.current_state),
        opening_mode: OpeningMode::SingleRealized,
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
        t,
        &prep_params,
    )
    .map_err(|e| SimulationError::SolvePrep(e.to_string()))?;
    // stage_id (the commissioning key the dormancy predicate compares NCS windows
    // against), NOT the stage index `t`: they differ when negative-id placeholder
    // stages are filtered out.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let stage_id = training_ctx
        .stages
        .get(t)
        .map_or(t as i32, |stage| stage.id);
    let n_stochastic_ncs = stochastic.n_stochastic_ncs();

    // mem::take (capacity retained) so these can be filled from `view` slices tied
    // to `ws` while `&mut ws` is live; restored at function end for buffer reuse.
    let mut unscaled_primal: Vec<f64> = std::mem::take(&mut ws.scratch.unscaled_primal);
    let mut unscaled_dual: Vec<f64> = std::mem::take(&mut ws.scratch.unscaled_dual);

    let col_scale = &ctx.templates[ids.t].col_scale;
    let row_scale = &ctx.templates[ids.t].row_scale;

    let view_objective: f64 = if let Some(params) = dcs {
        // Simulation has no iteration counter; seed with `current_iteration = 0`.
        build_initial_resident_set(
            &fcf.pools[t],
            0,
            params.k2,
            &mut ws.backward_accum.dcs_initial_resident,
        );
        let dcs_ctx = DcsSolveContext {
            stage_index: t,
            scenario_index: ids.scenario_id as usize,
            iteration: None, // disables the k1 window → every cut a candidate
            // Simulation solves one LP per (stage, scenario): always fresh.
            continue_carry: false,
        };
        // Disjoint borrows of `ws`: `solver`, `dcs_initial_resident` (shared), and
        // `dcs_solve` (mut) are distinct fields.
        lazy_solve_preloaded(
            &mut ws.solver,
            &ctx.templates[t],
            &fcf.pools[t],
            state,
            &training_ctx.cut_state_layouts[t],
            col_scale,
            None,
            &ws.backward_accum.dcs_initial_resident,
            &params,
            &mut ws.backward_accum.dcs_solve,
            dcs_ctx,
        )
        .map_err(|e| map_sim_solver_error(e, ids))?;
        let view = ws.backward_accum.dcs_solve.result_view();
        let objective = view.objective;
        fill_unscaled(&mut unscaled_primal, view.primal, col_scale);
        // INVARIANT: on the DCS path `view.dual` is LONGER than the structural
        // template (the lazy loop appends resident cut rows), carrying cut-row
        // duals at indices `>= template_num_rows`. Harmless: the reader only reads
        // structural-row indices. Do NOT truncate or add a `dual.len() ==
        // template_num_rows` check — that holds on the frozen path but NOT here, and
        // would drop the structural duals the reader needs.
        fill_unscaled_dual(&mut unscaled_dual, view.dual, row_scale);
        let _ = view;
        objective
    } else {
        let inputs = StageInputs {
            stage_context: ctx,
            pool: &fcf.pools[t],
            stored_basis: load_spec.warm_basis,
            stage_index: t,
            scenario_index: ids.scenario_id as usize,
            iteration: None, // simulation has no iteration counter
        };

        let view = run_stage_solve(ws, &inputs).map_err(|e| map_sim_solver_error(e, ids))?;

        let objective = view.objective;
        fill_unscaled(&mut unscaled_primal, view.primal, col_scale);
        fill_unscaled_dual(&mut unscaled_dual, view.dual, row_scale);
        let _ = view;
        objective
    };

    ws.scratch.unscaled_primal = unscaled_primal;
    ws.scratch.unscaled_dual = unscaled_dual;

    let (immediate_cost, result) = extract_sim_stage_result(
        &mut ws.scratch.inflow_m3s_buf,
        &mut ws.scratch.row_lower_buf,
        &ws.scratch.load_rhs_buf,
        &ws.scratch.ncs_col_upper_buf,
        &mut ws.scratch.ncs_col_upper_extract_buf,
        &ws.scratch.unscaled_primal,
        &ws.scratch.unscaled_dual,
        view_objective,
        ctx,
        output,
        state,
        study_dims,
        ids,
        stage_id,
        n_stochastic_ncs,
        lookups,
    );
    // Snapshot incoming lags before state is overwritten below.
    let lag_start = state.inflow_lags.start;
    let lag_len = state.hydro_count * state.max_par_order;
    ws.scratch.lag_matrix_buf.clear();
    ws.scratch
        .lag_matrix_buf
        .extend_from_slice(&ws.current_state[lag_start..lag_start + lag_len]);

    ws.current_state.clear();
    ws.current_state
        .extend_from_slice(&ws.scratch.unscaled_primal[..state.n_state]);

    let stage_lag = ctx
        .stage_lag_transitions
        .get(ids.t)
        .copied()
        .unwrap_or(StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: true,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: false,
        });
    let downstream_par_order = ws
        .scratch
        .downstream_completed_lags
        .len()
        .checked_div(ws.scratch.lag_accumulator.len())
        .unwrap_or(0);
    // Pass unscaled_primal as a separate borrow so the borrow checker sees it is
    // disjoint from the &mut ws.scratch.lag_* fields passed alongside it.
    let unscaled_primal_ref: &[f64] = &ws.scratch.unscaled_primal;
    accumulate_and_shift_lag_state(
        &mut ws.current_state,
        &ws.scratch.lag_matrix_buf,
        unscaled_primal_ref,
        state,
        &stage_lag,
        &mut LagAccumState {
            accumulator: &mut ws.scratch.lag_accumulator,
            weight_accum: &mut ws.scratch.lag_weight_accum,
        },
        &mut DownstreamAccumState {
            accumulator: &mut ws.scratch.downstream_accumulator,
            weight_accum: &mut ws.scratch.downstream_weight_accum,
            completed_lags: &mut ws.scratch.downstream_completed_lags,
            n_completed: &mut ws.scratch.downstream_n_completed,
            par_order: downstream_par_order,
        },
    );
    debug_assert_bucket_copy_gap_intact(&ws.current_state, unscaled_primal_ref, state);

    Ok((immediate_cost, result))
}

/// Extract the cost and result record from a solved simulation stage LP.
///
/// RATIONALE (`too_many_arguments`): takes individual scratch field borrows rather
/// than `&mut ScratchBuffers` so `unscaled_primal`/`unscaled_dual` can be passed as
/// `&[f64]` while `inflow_m3s_buf`/`row_lower_buf` are `&mut`; a single
/// `&mut ScratchBuffers` would block that disjoint split.
// Rationale (too_many_lines): a single linear per-stage assembly culminating in one
// `StageExtractionSpec` literal; splitting it would only relocate the ~25 borrowed
// inputs into a parameter list, scattering the assembly the literal reads.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn extract_sim_stage_result(
    inflow_m3s_buf: &mut Vec<f64>,
    row_lower_buf: &mut Vec<f64>,
    load_rhs_buf: &[f64],
    ncs_col_upper_buf: &[f64],
    ncs_col_upper_extract_buf: &mut Vec<f64>,
    unscaled_primal: &[f64],
    unscaled_dual: &[f64],
    view_objective: f64,
    ctx: &StageContext<'_>,
    output: &SimulationOutputSpec<'_>,
    state: &StateSpace,
    study_dims: &StudyDimensions,
    ids: &SimStageIds,
    stage_id: i32,
    n_stochastic_ncs: usize,
    lookups: &SimLookups,
) -> (f64, SimulationStageResult) {
    let t = ids.t;
    let theta_obj_coeff = ctx
        .templates
        .get(t)
        .and_then(|tmpl| tmpl.objective.get(state.theta).copied())
        .unwrap_or(1.0);
    let theta_contribution = unscaled_primal[state.theta] * theta_obj_coeff;
    let immediate_cost = (view_objective - theta_contribution) * ctx.cost_scale_factor;
    // Realized inflow Z_t from the z_h primal: total natural inflow (PAR lag
    // included), gross of withdrawal.
    inflow_m3s_buf.clear();
    for h in 0..ctx.n_hydros {
        inflow_m3s_buf.push(unscaled_primal[state.z_inflow.start + h]);
    }
    let blk_hrs = output
        .block_hours_per_stage
        .get(t)
        .map_or(&[][..], |v| v.as_slice());
    let (load_row_start, load_n_blks) = if ctx.n_load_buses > 0 {
        (
            ctx.load_balance_row_starts[t],
            ctx.block_counts_per_stage[t],
        )
    } else {
        (0, 0)
    };
    let row_lower_ref = build_row_lower_unscaled(
        &ctx.templates[t].row_lower,
        &ctx.templates[t].row_scale,
        load_rhs_buf,
        row_lower_buf,
        ctx.n_load_buses,
        load_row_start,
        load_n_blks,
        ctx.load_bus_indices,
    );
    // NCS upper bounds for extraction, in dense system-column order
    // (`ncs_sys * stage_n_blks + blk`).
    let ncs_n = output.n_ncs;
    let ncs_col_start = output.ncs_col_starts.get(t).copied().unwrap_or(0);
    let stage_n_blks = ctx.block_counts_per_stage.get(t).copied().unwrap_or(0);
    // Pumping-flow column base for this stage; `StageLayout` is its sole owner.
    let pumping_col_start = output.pumping_col_starts.get(t).copied().unwrap_or(0);
    let n_pumping = output.n_pumping;
    // Per-stage geometry: a single global stage-0 geometry would mis-stride any
    // stage with a differing block count.
    debug_assert!(
        output.geometry_per_stage.is_empty()
            || output.geometry_per_stage.len() == ctx.templates.len(),
        "geometry_per_stage must carry one entry per study stage when populated",
    );
    let geometry_default = StageGeometry::default();
    let geometry = output
        .geometry_per_stage
        .get(t)
        .unwrap_or(&geometry_default);
    // Start from the template `col_upper`, then overwrite each non-dormant
    // stochastic column with the per-scenario realized availability. A dormant slot
    // is skipped so its template `0` survives — copying its stochastic cap would
    // report a nonzero available for a column the LP pinned to `0`.
    let ncs_col_upper: &[f64] = if ncs_n > 0 && stage_n_blks > 0 {
        let start = ncs_col_start;
        let end = start + ncs_n * stage_n_blks;
        ncs_col_upper_extract_buf.clear();
        if end <= ctx.templates[t].col_upper.len() {
            ncs_col_upper_extract_buf.extend_from_slice(&ctx.templates[t].col_upper[start..end]);
        } else {
            ncs_col_upper_extract_buf.resize(ncs_n * stage_n_blks, 0.0);
        }
        if n_stochastic_ncs > 0 && !ncs_col_upper_buf.is_empty() {
            let dense_col = ctx.ncs_stochastic_dense_col;
            let windows = ctx.ncs_stochastic_windows;
            for (slot, &col) in dense_col.iter().enumerate() {
                let (entry, exit) = windows[slot];
                if !commissioning_active(entry, exit, stage_id) {
                    continue;
                }
                let dst = col * stage_n_blks;
                let src = slot * stage_n_blks;
                debug_assert!(
                    dst + stage_n_blks <= ncs_col_upper_extract_buf.len(),
                    "ncs extract dst out of range: dense_col < ncs_n by construction",
                );
                debug_assert!(
                    src + stage_n_blks <= ncs_col_upper_buf.len(),
                    "ncs extract src out of range: slot < n_stochastic_ncs by construction",
                );
                ncs_col_upper_extract_buf[dst..dst + stage_n_blks]
                    .copy_from_slice(&ncs_col_upper_buf[src..src + stage_n_blks]);
            }
        }
        ncs_col_upper_extract_buf.as_slice()
    } else {
        &[]
    };
    // Per-stage hydro FPHA/evap lookup (membership is per-`(hydro, stage)`). The
    // empty-table fallback is sized to `hydro_ids.len()` so `lookup.fpha[h]` stays
    // in bounds; built only on that path, never on the production hot path.
    let hydro_lookup_default;
    let hydro_lookup = if let Some(l) = lookups.hydro_per_stage.get(t) {
        l
    } else {
        hydro_lookup_default = HydroReverseLookup::build(
            &StageGeometry::default(),
            output.entity_counts.hydro_ids.len(),
        );
        &hydro_lookup_default
    };
    let result = extract_stage_result_with_lookups(
        &SolutionView {
            primal: unscaled_primal,
            dual: unscaled_dual,
            objective: view_objective,
            objective_coeffs: &ctx.templates[t].objective,
            row_lower: row_lower_ref,
        },
        &StageExtractionSpec {
            state,
            study_dims,
            n_blks: stage_n_blks,
            geometry,
            entity_counts: output.entity_counts,
            inflow_m3s_per_hydro: inflow_m3s_buf,
            block_hours: blk_hrs,
            generic_constraint_entries: output
                .generic_constraint_row_entries
                .get(t)
                .map_or(&[], Vec::as_slice),
            ncs_col_start,
            n_ncs: ncs_n,
            ncs_entity_ids: output
                .ncs_entity_ids_per_stage
                .get(t)
                .map_or(&[], Vec::as_slice),
            ncs_col_upper,
            pumping_col_start,
            n_pumping,
            pumping_consumption_mw_per_m3s: output.pumping_consumption_mw_per_m3s,
            contract_prices: output
                .contract_prices_per_stage
                .get(t)
                .map_or(&[], Vec::as_slice),
            contract_is_import: output.contract_is_import,
            diversion_upstream: output.diversion_upstream,
            hydro_productivities: output
                .hydro_productivities_per_stage
                .get(t)
                .map_or(&[], Vec::as_slice),
            col_scale: &ctx.templates[t].col_scale,
            row_scale: &ctx.templates[t].row_scale,
            cumulative_discount_factor: ctx
                .cumulative_discount_factors
                .get(t)
                .copied()
                .unwrap_or(1.0),
            cost_scale_factor: ctx.cost_scale_factor,
            energy_conversion: output.energy_conversion,
            hydro_min_storage_hm3: output.hydro_min_storage_hm3,
            stage_index: t,
            n_stages: ctx.templates.len(),
            anticipated_windows: ctx.anticipated_windows,
            study_stage_ids: ctx.study_stage_ids,
        },
        ids.stage_id_u32,
        hydro_lookup,
        &lookups.thermal,
    );
    (immediate_cost, result)
}

/// Reset workspace state to the initial conditions for a new scenario.
fn reset_scenario_state<S: SolverInterface>(
    ws: &mut SolverWorkspace<S>,
    sampler: &ForwardSampler<'_>,
    global_scenario: u32,
    total_scenarios: u32,
    inflow_lags_start: usize,
    training_ctx: &TrainingContext<'_>,
) {
    // Reset solver simplex state at the scenario boundary so a scenario's result
    // cannot depend on which scenarios ran before it (determinism across thread/
    // rank counts). No-op for HiGHS; recreates the model for CLP, whose
    // `Clp_loadProblem` leaves the rim/pricing state stale.
    ws.solver.reset_solver_state();

    let TrainingContext {
        initial_state,
        lag_accum_seed,
        lag_weight_seed,
        ..
    } = training_ctx;
    ws.current_state.clear();
    ws.current_state.extend_from_slice(initial_state);
    sampler.apply_initial_state(
        &ClassSampleRequest {
            iteration: 0,
            scenario: global_scenario,
            stage: 0,
            stage_idx: 0,
            total_scenarios,
            noise_group_id: 0,
        },
        &mut ws.current_state,
        inflow_lags_start,
    );
    // Seed (or zero) the lag accumulator so it does not carry state across
    // scenarios; a non-empty seed pre-fills the partial period with pre-study data.
    if lag_accum_seed.is_empty() {
        ws.scratch.lag_accumulator.fill(0.0);
        ws.scratch.lag_weight_accum.fill(0.0);
    } else {
        ws.scratch.lag_accumulator[..lag_accum_seed.len()].copy_from_slice(lag_accum_seed);
        ws.scratch.lag_weight_accum[..lag_weight_seed.len()].copy_from_slice(lag_weight_seed);
    }
    ws.scratch.downstream_accumulator.fill(0.0);
    ws.scratch.downstream_weight_accum = 0.0;
    ws.scratch.downstream_completed_lags.fill(0.0);
    ws.scratch.downstream_n_completed = 0;
}

pub(crate) fn process_scenario_stages<S: SolverInterface>(
    ws: &mut SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    fcf: &FutureCostFunction,
    training_ctx: &TrainingContext<'_>,
    load_spec: &SimScenarioLoadSpec<'_>,
    output: &SimulationOutputSpec<'_>,
    ids: &mut ScenarioIds<'_>,
    lookups: &SimLookups,
) -> Result<(f64, Vec<SimulationStageResult>), SimulationError> {
    let TrainingContext { state, .. } = training_ctx;
    reset_scenario_state(
        ws,
        ids.sampler,
        ids.global_scenario,
        ids.total_scenarios,
        state.inflow_lags.start,
        training_ctx,
    );
    let mut total_cost = 0.0_f64;
    let mut stage_results = Vec::with_capacity(ids.num_stages);

    #[allow(clippy::needless_range_loop)] // t indexes load_spec, ctx arrays, and SimStageIds
    for t in 0..ids.num_stages {
        #[allow(clippy::cast_possible_truncation)]
        let stage_id_u32 = t as u32;
        let noise = ids.sampler.sample(SampleRequest {
            iteration: 0,
            scenario: ids.global_scenario,
            stage: stage_id_u32,
            stage_idx: t,
            noise_buf: ids.raw_noise_buf,
            perm_scratch: ids.perm_scratch,
            total_scenarios: ids.total_scenarios,
            noise_group_id: ctx.noise_group_id_at(t),
        })?;
        let raw_noise = noise.as_slice();

        let (cost, result) = solve_simulation_stage(
            ws,
            ctx,
            fcf,
            training_ctx,
            &load_spec.stage(t),
            output,
            &SimStageIds {
                t,
                stage_id_u32,
                scenario_id: ids.scenario_id,
            },
            lookups,
            raw_noise,
        )?;
        let cum_d = ctx
            .cumulative_discount_factors
            .get(t)
            .copied()
            .unwrap_or(1.0);
        total_cost += cum_d * cost;
        stage_results.push(result);
    }
    Ok((total_cost, stage_results))
}

/// Emit an in-progress simulation event if a sender is available.
pub(crate) fn emit_sim_progress(
    sender: Option<&Sender<TrainingEvent>>,
    scenario_cost: f64,
    solve_time_ms: f64,
    lp_solves: u64,
    completed: u32,
    total: u32,
    elapsed_ms: u64,
) {
    if let Some(s) = sender {
        let _ = s.send(TrainingEvent::SimulationProgress {
            scenarios_complete: completed,
            scenarios_total: total,
            elapsed_ms,
            scenario_cost,
            solve_time_ms,
            lp_solves,
        });
    }
}

/// Accumulate per-category costs from all stage results, send the scenario
/// result through the channel, and return a compact `(scenario_id, total_cost,
/// category_costs)` tuple for MPI aggregation.
pub(crate) fn dispatch_scenario_result(
    output: &SimulationOutputSpec<'_>,
    scenario_id: u32,
    total_cost: f64,
    stage_results: Vec<SimulationStageResult>,
) -> Result<(u32, f64, ScenarioCategoryCosts), SimulationError> {
    let mut category_costs = ScenarioCategoryCosts {
        resource_cost: 0.0,
        recourse_cost: 0.0,
        violation_cost: 0.0,
        regularization_cost: 0.0,
        imputed_cost: 0.0,
    };
    for sr in &stage_results {
        for c in &sr.costs {
            accumulate_category_costs(c, &mut category_costs);
        }
    }
    let compact_category = category_costs.clone();
    output
        .result_tx
        .send(SimulationScenarioResult {
            scenario_id,
            total_cost,
            per_category_costs: category_costs,
            stages: stage_results,
        })
        .map_err(|_| SimulationError::ChannelClosed)?;
    Ok((scenario_id, total_cost, compact_category))
}

/// Evaluate the trained SDDP policy on a set of scenarios.
///
/// Thin shim that delegates to `SimulationState::run`.
/// All scheduling, freeze logic, and rayon parallelism live in
/// `crate::simulation::state`.
///
/// # Errors
///
/// Returns `Err(SimulationError::LpInfeasible { .. })` when a stage LP has no
/// feasible solution, `Err(SimulationError::SolverError { .. })` for other
/// terminal LP solver failures, and `Err(SimulationError::ChannelClosed)` when
/// the channel receiver has been dropped.
pub fn simulate<S, C: Communicator>(
    workspaces: &mut [SolverWorkspace<S>],
    ctx: &StageContext<'_>,
    fcf: &FutureCostFunction,
    training_ctx: &TrainingContext<'_>,
    config: &SimulationConfig,
    output: SimulationOutputSpec<'_>,
    frozen_templates: Option<&[StageTemplate]>,
    stage_bases: &[Option<CapturedBasis>],
    comm: &C,
) -> Result<SimulationRunResult, SimulationError>
where
    S: SolverInterface<Profile = ActiveProfile> + Send,
{
    use crate::simulation::state::{SimulationInputs, SimulationState};
    let mut state = SimulationState::new(training_ctx.horizon.num_stages());
    state.set_profile(config.profile);
    state.run(&mut SimulationInputs::new(
        workspaces,
        ctx,
        fcf,
        training_ctx,
        config,
        output,
        frozen_templates,
        stage_bases,
        comm,
    ))
}

#[cfg(test)]
mod tests;
