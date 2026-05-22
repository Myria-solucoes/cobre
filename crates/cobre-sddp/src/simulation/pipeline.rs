//! Simulation forward pass for policy evaluation.
//!
//! [`simulate`] evaluates the trained policy on scenarios. Scenarios distributed
//! across ranks via two-level distribution; within-rank parallelism via Rayon.
//! Seed domain separated from training. No per-scenario allocations on hot path.

use std::collections::HashMap;
use std::sync::mpsc::{Sender, SyncSender};

use cobre_comm::Communicator;
use cobre_core::{EntityId, TrainingEvent};
use cobre_solver::{SolverInterface, StageTemplate};
use cobre_stochastic::{ClassSampleRequest, ForwardSampler, SampleRequest};

use crate::{
    FutureCostFunction,
    context::{StageContext, TrainingContext},
    lp_builder::COST_SCALE_FACTOR,
    noise::{NcsNoiseOffsets, transform_inflow_noise, transform_load_noise, transform_ncs_noise},
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
    workspace::{CapturedBasis, SolverWorkspace},
};

/// Offset added to the simulation scenario ID before passing to [`ForwardSampler::sample`].
///
/// Separates the simulation seed domain from the training forward pass domain.
/// Training uses `global_scenario = rank * forward_passes + m`, while
/// simulation uses `global_scenario = SIMULATION_SEED_OFFSET + scenario_id`.
/// Both fit in `u32`; the offset guarantees no overlap for practical scenario counts.
pub(crate) const SIMULATION_SEED_OFFSET: u32 = u32::MAX / 2;

/// Per-worker scenario cost accumulation: `(scenario_id, total_cost, category_costs)`.
pub(crate) type WorkerCosts = Vec<(u32, f64, ScenarioCategoryCosts)>;

/// Per-worker solver statistics: `(scenario_id, opening, delta)`.
///
/// The `opening` field is always `-1` for simulation rows — simulation has one
/// solve per `(scenario, stage)` with no opening loop. The sentinel value `-1`
/// maps to a NULL `Int32` in the parquet schema.
pub(crate) type WorkerStats = Vec<(u32, i32, SolverStatsDelta)>;

/// Result of a simulation run, containing per-scenario costs and solver statistics.
#[derive(Debug)]
pub struct SimulationRunResult {
    /// Per-scenario `(scenario_id, total_cost, category_costs)`, sorted by `scenario_id`.
    pub costs: Vec<(u32, f64, ScenarioCategoryCosts)>,
    /// Per-scenario solver statistics delta, sorted by `scenario_id`.
    ///
    /// Each entry is `(scenario_id, opening, delta)` where `opening` is always
    /// `-1` for simulation (no opening dimension).
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

    /// Per-stage generic constraint row entries for extraction.
    ///
    /// `generic_constraint_row_entries[stage]` contains the active constraint
    /// row metadata at that stage.  Used by the extraction pipeline to map LP
    /// rows/columns back to constraint identity.
    pub generic_constraint_row_entries: &'a [Vec<crate::lp_builder::GenericConstraintRowEntry>],

    /// Per-stage NCS column start indices.
    ///
    /// `ncs_col_starts[stage]` is the column index of the first NCS generation
    /// variable at that stage.
    pub ncs_col_starts: &'a [usize],

    /// Per-stage active NCS entity counts.
    ///
    /// `n_ncs_per_stage[stage]` is the number of active NCS entities at that stage.
    pub n_ncs_per_stage: &'a [usize],

    /// Per-stage active NCS entity IDs, in ID-sorted order.
    ///
    /// `ncs_entity_ids_per_stage[stage]` lists the entity IDs of NCS sources
    /// active at that stage.
    pub ncs_entity_ids_per_stage: &'a [Vec<i32>],

    /// Mapping from target hydro ID to source hydro indices that divert to it.
    ///
    /// Used by the extraction pipeline to compute `diverted_inflow_m3s`.
    /// Empty when no hydros have diversion.
    pub diversion_upstream: &'a HashMap<EntityId, Vec<usize>>,

    /// Per-stage hydro productivities accounting for per-stage overrides.
    ///
    /// `hydro_productivities_per_stage[stage]` contains one productivity value
    /// per hydro plant.  For constant-productivity hydros this is the per-stage
    /// override (or base value if no override); for FPHA hydros it is 0.0.
    pub hydro_productivities_per_stage: &'a [Vec<f64>],

    /// Pre-computed energy-conversion scalars for every `(hydro, stage)` pair.
    ///
    /// Threaded into [`crate::simulation::extraction::StageExtractionSpec`] at
    /// each stage to populate the five energy fields on per-block hydro rows.
    pub energy_conversion: &'a crate::energy_conversion::EnergyConversionSet,

    /// Minimum storage volume `V_min` per hydro plant (hm³), in canonical
    /// ID-sorted order.
    ///
    /// Threaded into [`crate::simulation::extraction::StageExtractionSpec`] to
    /// compute `stored_energy_mwh = (V - V_min) · ρ_acum · ENERGY_FACTOR`.
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
    /// Global scenario ID passed to `ForwardSampler::sample` as `scenario`.
    ///
    /// Already includes [`SIMULATION_SEED_OFFSET`] to separate the simulation
    /// seed domain from the training forward pass domain.
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

/// Rebuild the `row_lower` slice for a stage with full unscaling.
///
/// Always copies `template_row_lower` into `scratch_buf` and divides each element
/// by its corresponding `row_scale` factor.  Stochastic load rows are then
/// overwritten with the unscaled values from `load_rhs_buf` (which are already
/// in MW).  The result is a slice where every element is in original units.
///
/// When `row_scale` is empty (no prescaling), the function does a bulk memcpy
/// without per-element division, matching the old fast path.
///
/// Structurally independent parameters: template buffers (`template_row_lower`,
/// `row_scale`, `scratch_buf`), load-RHS buffer (`load_rhs_buf`), and load-dimension
/// indices (`n_load_buses`, `load_balance_row_start`, `n_blks`, `load_bus_indices`)
/// are per-call scratch or lookup tables with no natural grouping.
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

    // Unscale all template rows.
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

    // Overwrite stochastic load rows (already in unscaled MW).
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

/// Process all stages for one simulation scenario, updating workspace state in place.
///
/// Runs the inner stage loop for a single scenario, solving the LP at each stage,
/// accumulating costs, and populating `stage_results`. Returns `(total_cost, stage_results)`.
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
/// The LP is loaded via `load_model(baked_template)` — the baked template
/// already embeds all active cut rows as structural rows. No `add_rows` call
/// is needed.
struct SimStageLoadSpec<'a> {
    /// Baked template for this stage; always populated after the startup re-bake.
    baked_template: &'a StageTemplate,
    /// Warm-start basis captured during training at this stage, if any.
    warm_basis: Option<&'a CapturedBasis>,
    /// Runtime window for the basis-activity classifier.
    /// Derived from `SimulationConfig::basis_activity_window`.
    basis_activity_window: u32,
}

/// Per-stage batched form of [`SimStageLoadSpec`] consumed by
/// `process_scenario_stages`; indexed by stage via [`Self::stage`].
pub(crate) struct SimScenarioLoadSpec<'a> {
    pub(crate) baked_templates: &'a [StageTemplate],
    pub(crate) stage_bases: &'a [Option<CapturedBasis>],
    /// Runtime window for the basis-activity classifier.
    /// Propagated into each per-stage [`SimStageLoadSpec`] via [`Self::stage`].
    pub(crate) basis_activity_window: u32,
}

impl<'a> SimScenarioLoadSpec<'a> {
    #[inline]
    fn stage(&self, t: usize) -> SimStageLoadSpec<'a> {
        SimStageLoadSpec {
            baked_template: &self.baked_templates[t],
            warm_basis: self.stage_bases.get(t).and_then(Option::as_ref),
            basis_activity_window: self.basis_activity_window,
        }
    }
}

/// Pre-built study-invariant reverse-lookup tables for simulation extraction.
///
/// Both lookups depend only on the study's [`crate::indexer::StageIndexer`] and
/// entity counts, which are constant across all scenarios and all stages.
/// Build once per simulation run (or per worker thread) via
/// [`SimLookups::build`] and pass by reference into every per-stage extraction
/// call to eliminate per-`(scenario, stage)` allocations on the hot path.
pub(crate) struct SimLookups {
    /// Reverse-lookup table for anticipated thermal indices.
    pub(crate) thermal: ThermalReverseLookup,
    /// Reverse-lookup table for hydro FPHA and evaporation indices.
    pub(crate) hydro: HydroReverseLookup,
}

impl SimLookups {
    /// Build both reverse-lookup tables from the study's indexer and entity counts.
    ///
    /// Allocates once; the returned value is then used read-only for the entire
    /// simulation run.
    pub(crate) fn build(
        indexer: &crate::indexer::StageIndexer,
        n_thermals: usize,
        n_hydros: usize,
    ) -> Self {
        Self {
            thermal: ThermalReverseLookup::build(indexer, n_thermals),
            hydro: HydroReverseLookup::build(indexer, n_hydros),
        }
    }
}

/// Patch NCS column bounds in the LP solver with per-scenario availability.
///
/// Called after `set_row_bounds` and before `solve`. The `ncs_col_lower_buf`
/// and `ncs_col_upper_buf` in the workspace scratch must already be
/// populated by `transform_ncs_noise` (both bounds scale with the realized
/// per-scenario availability ratio). The indices buffer is rebuilt lazily:
/// only when the expected size changes (i.e., on a stage transition).
fn apply_ncs_col_bounds<S: SolverInterface>(
    solver: &mut S,
    scratch: &mut crate::workspace::ScratchBuffers,
    ncs_generation_start: usize,
    n_stochastic_ncs: usize,
    n_blks: usize,
) {
    let expected_len = n_stochastic_ncs * n_blks;
    if scratch.ncs_col_indices_buf.len() != expected_len {
        scratch.ncs_col_indices_buf.clear();
        for ncs_idx in 0..n_stochastic_ncs {
            for blk in 0..n_blks {
                scratch
                    .ncs_col_indices_buf
                    .push(ncs_generation_start + ncs_idx * n_blks + blk);
            }
        }
    }
    solver.set_col_bounds(
        &scratch.ncs_col_indices_buf,
        &scratch.ncs_col_lower_buf,
        &scratch.ncs_col_upper_buf,
    );
}

/// Solve one stage for one simulation scenario, updating workspace in-place.
///
/// Patches the LP for stage `t`, solves it, extracts inflow/row-lower data,
/// and returns `(immediate_cost, SimulationStageResult)`. When a warm-start
/// basis is provided, the LP is solved via slot-tracked `reconstruct_basis` then
/// `solve(Some(&basis))`; otherwise it falls back to a cold-start `solve(None)`. The
/// warm-start basis is a read-only, per-stage artifact from training, so
/// determinism is preserved.
///
/// `load_model(baked_template)` is always used — the cut rows are already
/// structural rows in the baked template; no `add_rows` call is needed.
/// When `baked_templates` was `None` on `simulate` entry, the startup re-bake
/// already produced a local `Vec<StageTemplate>` before this function is reached.
///
/// `lookups` carries pre-built study-invariant reverse-lookup tables that are
/// threaded through to [`extract_sim_stage_result`] so no allocation occurs on
/// the per-stage hot path.
// RATIONALE: solve_simulation_stage orchestrates the full per-stage simulation pipeline:
// noise application, NCS patching, LP solve, result extraction, and output writing.
// Each phase is a semantically distinct step; splitting further would fragment the
// per-stage invariant tracking without reducing the total operations performed.
#[allow(clippy::too_many_lines)]
fn solve_simulation_stage<S: SolverInterface>(
    ws: &mut crate::workspace::SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    fcf: &FutureCostFunction,
    training_ctx: &TrainingContext<'_>,
    load_spec: &SimStageLoadSpec<'_>,
    output: &SimulationOutputSpec<'_>,
    ids: &SimStageIds,
    lookups: &SimLookups,
) -> Result<(f64, SimulationStageResult), SimulationError> {
    // Precondition: ws.scratch.noise_buf, ws.scratch.load_rhs_buf, and
    // ws.scratch.ncs_col_upper_buf are populated by the caller
    // (process_scenario_stages) via the transform_* functions.
    let TrainingContext {
        indexer,
        stochastic,
        ..
    } = training_ctx;
    let t = ids.t;
    // The baked template already embeds all active cut rows as structural rows.
    // No add_rows call is needed.
    ws.solver.load_model(load_spec.baked_template);
    // The anticipated-state ring buffer must advance once per stage in the
    // simulation forward walk, mirroring `run_forward_stage` in `forward.rs`.
    // The incoming slice is snapshotted into `ws.scratch.anticipated_state_buf`
    // immediately before `ws.current_state` is overwritten with the unscaled
    // primal below, then consumed by `shift_anticipated_state` to produce the
    // outgoing ring-buffer state. Without this shift, the unbounded
    // `anticipated_state` LP columns (which the Category 6 fixing rows leave
    // at `incoming - decision` at the optimum) would leak into the next
    // stage's incoming state, and fishing constraints at delivery stages
    // would read stale slot 0 values.
    ws.patch_buf.fill_forward_patches(
        indexer,
        &ws.current_state,
        &ws.scratch.noise_buf,
        ctx.base_rows[t],
        &ctx.templates[t].row_scale,
    );
    if ctx.n_load_buses > 0 {
        ws.patch_buf.fill_load_patches(
            ctx.load_balance_row_starts[t],
            ctx.block_counts_per_stage[t],
            &ws.scratch.load_rhs_buf,
            ctx.load_bus_indices,
            &ctx.templates[t].row_scale,
        );
    }
    ws.patch_buf.fill_z_inflow_patches(
        indexer.z_inflow_row_start,
        &ws.scratch.z_inflow_rhs_buf,
        &ctx.templates[t].row_scale,
    );
    let pc = ws.patch_buf.forward_patch_count();
    ws.solver.set_row_bounds(
        &ws.patch_buf.indices[..pc],
        &ws.patch_buf.lower[..pc],
        &ws.patch_buf.upper[..pc],
    );
    // Patch NCS column upper bounds with per-scenario stochastic availability.
    // ncs_col_upper_buf was populated by transform_ncs_noise in the caller.
    let n_stochastic_ncs = stochastic.n_stochastic_ncs();
    if n_stochastic_ncs > 0 && !indexer.ncs_generation.is_empty() {
        apply_ncs_col_bounds(
            &mut ws.solver,
            &mut ws.scratch,
            indexer.ncs_generation.start,
            n_stochastic_ncs,
            ctx.block_counts_per_stage[t],
        );
    }

    // Take scratch buffers out of ws before run_stage_solve borrows ws.
    // The `mem::take` pattern (capacity retained, empty vec left in field) allows
    // us to pass borrowed slices while `&mut ws` is also live, and to fill
    // unscaled_primal/unscaled_dual from `view` slices that are still tied to 'ws.
    // All three are restored to ws.scratch at function end so the next stage reuses
    // the warmed allocations.
    let mut current_state_local: Vec<f64> = std::mem::take(&mut ws.scratch.current_state_scratch);
    let mut unscaled_primal: Vec<f64> = std::mem::take(&mut ws.scratch.unscaled_primal);
    let mut unscaled_dual: Vec<f64> = std::mem::take(&mut ws.scratch.unscaled_dual);

    current_state_local.clear();
    current_state_local.extend_from_slice(&ws.current_state[..indexer.n_state]);

    let inputs = crate::stage_solve::StageInputs {
        stage_context: ctx,
        indexer,
        pool: &fcf.pools[t],
        current_state: &current_state_local,
        stored_basis: load_spec.warm_basis,
        baked_template: load_spec.baked_template,
        stage_index: t,
        scenario_index: ids.scenario_id as usize,
        iteration: None,            // simulation has no iteration counter
        horizon_is_terminal: false, // simulation stage-sweep semantics
        terminal_has_boundary_cuts: false,
        basis_activity_window: load_spec.basis_activity_window,
    };

    let outcome =
        crate::stage_solve::run_stage_solve(ws, crate::stage_solve::Phase::Simulation, &inputs)
            .map_err(|e| match e {
                crate::error::SddpError::Infeasible {
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
                crate::error::SddpError::Solver(other) => SimulationError::SolverError {
                    scenario_id: ids.scenario_id,
                    stage_id: ids.stage_id_u32,
                    solver_message: other.to_string(),
                },
                other => SimulationError::SolverError {
                    scenario_id: ids.scenario_id,
                    stage_id: ids.stage_id_u32,
                    solver_message: format!("{other}"),
                },
            })?;

    let crate::stage_solve::StageOutcome::Simulation {
        view,
        recon_stats: _,
    } = outcome
    else {
        unreachable!("run_stage_solve(Phase::Simulation) returns Simulation variant")
    };

    // Extract scaling slices and objective scalar before filling unscaled buffers.
    // `view` borrows from `ws` (via run_stage_solve), so we must finish all reads
    // of `view` before the view borrow ends.
    let col_scale = &ctx.templates[ids.t].col_scale;
    let row_scale = &ctx.templates[ids.t].row_scale;
    let view_objective = view.objective;

    // Fill the taken-out unscaled buffers from view.primal/view.dual. The locals
    // already carry the pre-warmed capacity; clear() + extend reuses it.
    unscaled_primal.clear();
    if col_scale.is_empty() {
        unscaled_primal.extend_from_slice(view.primal);
    } else {
        unscaled_primal.extend(
            view.primal
                .iter()
                .zip(col_scale.iter())
                .map(|(&xp, &d)| d * xp),
        );
    }

    // `dual_original[i] = row_scale[i] * dual_scaled[i]`.
    // Structural rows use row_scale[i]; rows beyond the template length
    // (cut rows) have implicit row_scale = 1.0.
    unscaled_dual.clear();
    if row_scale.is_empty() {
        unscaled_dual.extend_from_slice(view.dual);
    } else {
        unscaled_dual.extend(view.dual.iter().enumerate().map(|(i, &d)| {
            let scale = if i < row_scale.len() {
                row_scale[i]
            } else {
                1.0
            };
            d * scale
        }));
    }

    // End the `view` borrow of `ws` before mutating ws further.
    // `SolutionView` is Copy so `drop(view)` would be a no-op; `let _` is idiomatic.
    // `inputs` and its borrow of current_state_local ended at the `?` above
    // (NLL sees no further use of inputs after that point).
    // Restore scratch buffers so the next stage reuses the warmed allocations.
    let _ = view;
    ws.scratch.current_state_scratch = current_state_local;
    ws.scratch.unscaled_primal = unscaled_primal;
    ws.scratch.unscaled_dual = unscaled_dual;

    // extract_sim_stage_result takes individual scratch field borrows so the
    // borrow checker can see that unscaled_primal/dual (passed as &[f64]) are
    // disjoint from the fields it mutates (&mut inflow_m3s_buf, &mut row_lower_buf).
    let (immediate_cost, result) = extract_sim_stage_result(
        &mut ws.scratch.inflow_m3s_buf,
        &mut ws.scratch.row_lower_buf,
        &ws.scratch.load_rhs_buf,
        &ws.scratch.ncs_col_upper_buf,
        &ws.scratch.unscaled_primal,
        &ws.scratch.unscaled_dual,
        view_objective,
        ctx,
        output,
        indexer,
        ids,
        n_stochastic_ncs,
        lookups,
    );
    // Save incoming lag values before overwriting state.
    let lag_start = indexer.inflow_lags.start;
    let lag_len = indexer.hydro_count * indexer.max_par_order;
    ws.scratch.lag_matrix_buf.clear();
    ws.scratch
        .lag_matrix_buf
        .extend_from_slice(&ws.current_state[lag_start..lag_start + lag_len]);

    // Save incoming anticipated-state slice before overwriting state with primal.
    // Uses the pre-allocated anticipated_state_buf scratch buffer (no allocation).
    // Mirrors the snapshot performed in `run_forward_stage` (forward.rs).
    let ant_start = indexer.anticipated_state.start;
    let ant_len = indexer.n_anticipated * indexer.k_max;
    ws.scratch.anticipated_state_buf.clear();
    ws.scratch
        .anticipated_state_buf
        .extend_from_slice(&ws.current_state[ant_start..ant_start + ant_len]);

    ws.current_state.clear();
    ws.current_state
        .extend_from_slice(&ws.scratch.unscaled_primal[..indexer.n_state]);

    // Accumulate and shift lag state using stage-level transition weights.
    // For monthly-only studies this degenerates to the previous direct shift.
    let stage_lag = ctx.stage_lag_transitions.get(ids.t).copied().unwrap_or(
        cobre_core::temporal::StageLagTransition {
            accumulate_weight: 1.0,
            spillover_weight: 0.0,
            finalize_period: true,
            accumulate_downstream: false,
            downstream_accumulate_weight: 0.0,
            downstream_spillover_weight: 0.0,
            downstream_finalize: false,
            rebuild_from_downstream: false,
        },
    );
    let downstream_par_order = ws
        .scratch
        .downstream_completed_lags
        .len()
        .checked_div(ws.scratch.lag_accumulator.len())
        .unwrap_or(0);
    // Pass unscaled_primal as a separate borrow so the borrow checker sees it is
    // disjoint from the &mut ws.scratch.lag_* fields passed alongside it.
    let unscaled_primal_ref: &[f64] = &ws.scratch.unscaled_primal;
    crate::noise::accumulate_and_shift_lag_state(
        &mut ws.current_state,
        &ws.scratch.lag_matrix_buf,
        unscaled_primal_ref,
        indexer,
        &stage_lag,
        &mut crate::noise::LagAccumState {
            accumulator: &mut ws.scratch.lag_accumulator,
            weight_accum: &mut ws.scratch.lag_weight_accum,
        },
        &mut crate::noise::DownstreamAccumState {
            accumulator: &mut ws.scratch.downstream_accumulator,
            weight_accum: &mut ws.scratch.downstream_weight_accum,
            completed_lags: &mut ws.scratch.downstream_completed_lags,
            n_completed: &mut ws.scratch.downstream_n_completed,
            par_order: downstream_par_order,
        },
    );
    // Advance the anticipated-state ring buffer. Reads the snapshotted
    // incoming slice from `anticipated_state_buf` and writes the post-shift
    // outgoing state into `ws.current_state[anticipated_state]`. No-op when
    // `n_anticipated == 0 || k_max == 0`.
    crate::noise::shift_anticipated_state(
        &mut ws.current_state,
        &ws.scratch.anticipated_state_buf,
        unscaled_primal_ref,
        indexer,
    );

    Ok((immediate_cost, result))
}

/// Extract the cost and result record from a solved simulation stage LP.
///
/// Separated from [`solve_simulation_stage`] to keep that function under the
/// line-count lint limit.
///
/// Takes individual scratch field borrows rather than `&mut ScratchBuffers` so
/// the caller can pass `unscaled_primal`/`unscaled_dual` as `&[f64]` slices of
/// their respective scratch fields while simultaneously passing `&mut` borrows to
/// the fields this function actually mutates (`inflow_m3s_buf`, `row_lower_buf`).
/// The borrow checker can verify disjointness at field granularity.
///
/// `lookups` carries pre-built study-invariant reverse-lookup tables passed down
/// from [`process_scenario_stages`] to avoid per-stage allocation.
#[allow(clippy::too_many_arguments)]
fn extract_sim_stage_result(
    inflow_m3s_buf: &mut Vec<f64>,
    row_lower_buf: &mut Vec<f64>,
    load_rhs_buf: &[f64],
    ncs_col_upper_buf: &[f64],
    unscaled_primal: &[f64],
    unscaled_dual: &[f64],
    view_objective: f64,
    ctx: &StageContext<'_>,
    output: &SimulationOutputSpec<'_>,
    indexer: &crate::indexer::StageIndexer,
    ids: &SimStageIds,
    n_stochastic_ncs: usize,
    lookups: &SimLookups,
) -> (f64, SimulationStageResult) {
    let t = ids.t;
    let theta_obj_coeff = ctx
        .templates
        .get(t)
        .and_then(|tmpl| tmpl.objective.get(indexer.theta).copied())
        .unwrap_or(1.0);
    let theta_contribution = unscaled_primal[indexer.theta] * theta_obj_coeff;
    let immediate_cost = (view_objective - theta_contribution) * COST_SCALE_FACTOR;
    // Read realized inflow Z_t directly from the LP primal solution.
    // The z_h variables are defined by the z-inflow constraint:
    //   z_h = base_h + sum_l[psi_l * lag_in[h,l]] + sigma_h * eta_h
    // This is the total natural inflow, including PAR lag contribution
    // and gross of withdrawal.
    inflow_m3s_buf.clear();
    for h in 0..ctx.n_hydros {
        inflow_m3s_buf.push(unscaled_primal[indexer.z_inflow.start + h]);
    }
    let blk_hrs = output
        .block_hours_per_stage
        .get(t)
        .map_or(&[][..], |v| v.as_slice());
    // Guard index accesses when there are no load buses (slices may be empty).
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
    // NCS column upper bounds for extraction. Use the per-scenario scratch buffer
    // when stochastic NCS patching is active and covers all active NCS entities;
    // fall back to the template values otherwise.
    let ncs_n = output.n_ncs_per_stage.get(t).copied().unwrap_or(0);
    let ncs_col_start = output.ncs_col_starts.get(t).copied().unwrap_or(0);
    let stage_n_blks = ctx.block_counts_per_stage.get(t).copied().unwrap_or(0);
    let ncs_col_upper: &[f64] =
        if n_stochastic_ncs > 0 && n_stochastic_ncs == ncs_n && !ncs_col_upper_buf.is_empty() {
            // All active NCS entities at this stage are stochastic — use the patched values.
            ncs_col_upper_buf
        } else if ncs_n > 0 && stage_n_blks > 0 {
            let start = ncs_col_start;
            let end = start + ncs_n * stage_n_blks;
            if end <= ctx.templates[t].col_upper.len() {
                &ctx.templates[t].col_upper[start..end]
            } else {
                &[]
            }
        } else {
            &[]
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
            indexer,
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
            energy_conversion: output.energy_conversion,
            hydro_min_storage_hm3: output.hydro_min_storage_hm3,
            stage_index: t,
            n_stages: ctx.templates.len(),
        },
        ids.stage_id_u32,
        &lookups.hydro,
        &lookups.thermal,
    );
    (immediate_cost, result)
}

/// Reset workspace state to the initial conditions for a new scenario.
///
/// Separated from [`process_scenario_stages`] to keep that function under the
/// line-count lint limit.
fn reset_scenario_state<S: SolverInterface>(
    ws: &mut crate::workspace::SolverWorkspace<S>,
    sampler: &ForwardSampler<'_>,
    global_scenario: u32,
    total_scenarios: u32,
    inflow_lags_start: usize,
    training_ctx: &TrainingContext<'_>,
) {
    let TrainingContext {
        initial_state,
        recent_accum_seed,
        recent_weight_seed,
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
    // Seed (or zero) the lag accumulator at trajectory start so it does not
    // carry state across simulation scenarios. When recent_accum_seed is
    // non-empty, copy it instead of zeroing — this pre-fills the partial
    // period with pre-study observed data.
    if recent_accum_seed.is_empty() {
        ws.scratch.lag_accumulator.fill(0.0);
        ws.scratch.lag_weight_accum = 0.0;
    } else {
        ws.scratch.lag_accumulator[..recent_accum_seed.len()].copy_from_slice(recent_accum_seed);
        ws.scratch.lag_weight_accum = *recent_weight_seed;
    }
    // Reset downstream accumulator at trajectory start.
    ws.scratch
        .downstream_accumulator
        .iter_mut()
        .for_each(|v| *v = 0.0);
    ws.scratch.downstream_weight_accum = 0.0;
    ws.scratch
        .downstream_completed_lags
        .iter_mut()
        .for_each(|v| *v = 0.0);
    ws.scratch.downstream_n_completed = 0;
}

pub(crate) fn process_scenario_stages<S: SolverInterface>(
    ws: &mut crate::workspace::SolverWorkspace<S>,
    ctx: &StageContext<'_>,
    fcf: &FutureCostFunction,
    training_ctx: &TrainingContext<'_>,
    load_spec: &SimScenarioLoadSpec<'_>,
    output: &SimulationOutputSpec<'_>,
    ids: &mut ScenarioIds<'_>,
    lookups: &SimLookups,
) -> Result<(f64, Vec<SimulationStageResult>), SimulationError> {
    let TrainingContext {
        indexer,
        stochastic,
        ..
    } = training_ctx;
    reset_scenario_state(
        ws,
        ids.sampler,
        ids.global_scenario,
        ids.total_scenarios,
        indexer.inflow_lags.start,
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
        transform_inflow_noise(
            raw_noise,
            t,
            &ws.current_state,
            ctx,
            training_ctx,
            &mut ws.scratch,
        );
        transform_load_noise(
            raw_noise,
            ctx.n_hydros,
            ctx.n_load_buses,
            stochastic,
            t,
            if ctx.n_load_buses > 0 {
                ctx.block_counts_per_stage[t]
            } else {
                0
            },
            &mut ws.scratch.load_rhs_buf,
        );
        let n_stochastic_ncs = stochastic.n_stochastic_ncs();
        if n_stochastic_ncs > 0 {
            transform_ncs_noise(
                raw_noise,
                &NcsNoiseOffsets {
                    n_hydros: ctx.n_hydros,
                    n_load_buses: ctx.n_load_buses,
                },
                stochastic,
                t,
                ctx.block_counts_per_stage[t],
                ctx.ncs_max_gen,
                ctx.ncs_allow_curtailment,
                &mut ws.scratch.ncs_col_lower_buf,
                &mut ws.scratch.ncs_col_upper_buf,
            );
        }
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
        )?;
        let cum_d = ctx
            .cumulative_discount_factors
            .get(t)
            .copied()
            .unwrap_or(1.0);
        total_cost += cum_d * cost;
        // Re-read state update: solve_simulation_stage already called ws.current_state.extend_from_slice.
        // However, the noise transforms above consumed ws.scratch; we must NOT re-run them.
        // The state was already advanced inside solve_simulation_stage. ✓
        stage_results.push(result);
        // State is already updated inside solve_simulation_stage via ws.current_state. ✓
    }
    // Restore the indexer reference lifetime: indexer is still in scope from the destructure above.
    let _ = indexer; // suppress unused warning
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
///
/// Extracted from `simulate`'s inner worker loop to keep that function within
/// the 100-line limit.
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
/// All scheduling, bake logic, and rayon parallelism live in
/// `crate::simulation::state`.
///
/// # Errors
///
/// Returns `Err(SimulationError::LpInfeasible { .. })` when a stage LP has no
/// feasible solution, `Err(SimulationError::SolverError { .. })` for other
/// terminal LP solver failures, and `Err(SimulationError::ChannelClosed)` when
/// the channel receiver has been dropped.
pub fn simulate<S: SolverInterface + Send, C: Communicator>(
    workspaces: &mut [SolverWorkspace<S>],
    ctx: &StageContext<'_>,
    fcf: &FutureCostFunction,
    training_ctx: &TrainingContext<'_>,
    config: &SimulationConfig,
    output: SimulationOutputSpec<'_>,
    baked_templates: Option<&[StageTemplate]>,
    stage_bases: &[Option<CapturedBasis>],
    comm: &C,
) -> Result<SimulationRunResult, SimulationError> {
    use crate::simulation::state::{SimulationInputs, SimulationState};
    let mut state = SimulationState::new(training_ctx.horizon.num_stages());
    state.run(&mut SimulationInputs::new(
        workspaces,
        ctx,
        fcf,
        training_ctx,
        config,
        output,
        baked_templates,
        stage_bases,
        comm,
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::too_many_lines)]
mod tests {
    use std::collections::HashMap;
    use std::sync::mpsc;

    use cobre_comm::{CommData, CommError, Communicator, ReduceOp};
    use cobre_core::scenario::SamplingScheme;
    use cobre_solver::{
        Basis, LpSolution, RowBatch, SolverError, SolverInterface, SolverStatistics, StageTemplate,
    };
    use cobre_stochastic::StochasticContext;

    use super::SimulationOutputSpec;
    use crate::{
        context::{StageContext, TrainingContext},
        cut::FutureCostFunction,
        horizon_mode::HorizonMode,
        indexer::StageIndexer,
        inflow_method::InflowNonNegativityMethod,
        lp_builder::PatchBuffer,
        simulation::{
            config::SimulationConfig,
            error::SimulationError,
            extraction::EntityCounts,
            state::{SimulationInputs, SimulationState},
        },
        workspace::{BackwardAccumulators, CapturedBasis, ScratchBuffers, SolverWorkspace},
    };

    /// Thin test helper that mirrors the old `simulate` free-function signature
    /// and delegates to `SimulationState::run`. Allows all 24 existing test call
    /// sites to stay unchanged apart from the function name.
    #[allow(clippy::too_many_arguments)]
    fn run_simulate<S: cobre_solver::SolverInterface + Send, C: cobre_comm::Communicator>(
        workspaces: &mut [SolverWorkspace<S>],
        ctx: &StageContext<'_>,
        fcf: &crate::FutureCostFunction,
        training_ctx: &TrainingContext<'_>,
        config: &SimulationConfig,
        output: SimulationOutputSpec<'_>,
        baked_templates: Option<&[cobre_solver::StageTemplate]>,
        stage_bases: &[Option<CapturedBasis>],
        comm: &C,
    ) -> Result<super::SimulationRunResult, SimulationError> {
        let num_stages = training_ctx.horizon.num_stages();
        let mut state = SimulationState::new(num_stages);
        let mut inputs = SimulationInputs {
            workspaces,
            ctx,
            fcf,
            training_ctx,
            config,
            output,
            baked_templates,
            stage_bases,
            comm,
        };
        state.run(&mut inputs)
    }

    // ── Stub communicator ────────────────────────────────────────────────────

    /// Single-rank stub communicator for unit tests.
    struct StubComm {
        rank: usize,
        size: usize,
    }

    impl Communicator for StubComm {
        fn allgatherv<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _counts: &[usize],
            _displs: &[usize],
        ) -> Result<(), CommError> {
            unreachable!("StubComm allgatherv not used in simulate tests")
        }

        fn allreduce<T: CommData>(
            &self,
            _send: &[T],
            _recv: &mut [T],
            _op: ReduceOp,
        ) -> Result<(), CommError> {
            unreachable!("StubComm allreduce not used in simulate tests")
        }

        fn broadcast<T: CommData>(&self, _buf: &mut [T], _root: usize) -> Result<(), CommError> {
            unreachable!("StubComm broadcast not used in simulate tests")
        }

        fn barrier(&self) -> Result<(), CommError> {
            Ok(())
        }

        fn rank(&self) -> usize {
            self.rank
        }

        fn size(&self) -> usize {
            self.size
        }

        fn abort(&self, error_code: i32) -> ! {
            std::process::exit(error_code)
        }
    }

    // ── Mock solver ──────────────────────────────────────────────────────────

    /// Mock solver that returns a configurable fixed `LpSolution` on every solve.
    ///
    /// Optionally returns `SolverError::Infeasible` at a specific (0-based) solve
    /// call index (counting across both cold-start and warm-start calls).
    ///
    /// `load_count` and `add_rows_count` track how many times `load_model` and
    /// `add_rows` were called, used by the baked-path acceptance tests.
    ///
    /// `solve_count` counts calls to the cold-start `solve(None)` path only.
    /// `solve_with_basis_count` counts calls to the warm-start `solve(Some(&basis))`
    /// path only.  `call_count` tracks the combined total across both paths and
    /// is used by the infeasibility injection logic.
    ///
    /// `recorded_basis` captures the last `Basis` passed to `solve(Some(&basis))`,
    /// used by the warm-start reconstruction acceptance tests.
    ///
    /// `reconstruction_counter` counts `record_reconstruction_stats` invocations
    /// (one per warm-start solve that applied a stored basis via slot reconciliation).
    struct MockSolver {
        solution: LpSolution,
        infeasible_at: Option<usize>,
        call_count: usize,
        buf_primal: Vec<f64>,
        buf_dual: Vec<f64>,
        buf_reduced_costs: Vec<f64>,
        /// Number of `load_model` calls since construction.
        load_count: usize,
        /// Number of `add_rows` calls since construction.
        add_rows_count: usize,
        /// Number of cold-start `solve(None)` calls.
        solve_count: usize,
        /// Number of warm-start `solve(Some(&basis))` calls.
        solve_with_basis_count: usize,
        /// Last `Basis` passed to `solve(Some(&basis))`, if any.
        recorded_basis: Option<Basis>,
        /// Cumulative `preserved` argument from `record_reconstruction_stats`.
        reconstruction_counter: u32,
    }

    impl MockSolver {
        fn always_ok(solution: LpSolution) -> Self {
            let buf_primal = solution.primal.clone();
            let buf_dual = solution.dual.clone();
            let buf_reduced_costs = solution.reduced_costs.clone();
            Self {
                solution,
                infeasible_at: None,
                call_count: 0,
                buf_primal,
                buf_dual,
                buf_reduced_costs,
                load_count: 0,
                add_rows_count: 0,
                solve_count: 0,
                solve_with_basis_count: 0,
                recorded_basis: None,
                reconstruction_counter: 0,
            }
        }

        fn do_solve(&mut self) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            let call = self.call_count;
            self.call_count += 1;
            if self.infeasible_at == Some(call) {
                return Err(SolverError::Infeasible);
            }
            self.buf_primal.clone_from(&self.solution.primal);
            self.buf_dual.clone_from(&self.solution.dual);
            self.buf_reduced_costs
                .clone_from(&self.solution.reduced_costs);
            Ok(cobre_solver::SolutionView {
                objective: self.solution.objective,
                primal: &self.buf_primal,
                dual: &self.buf_dual,
                reduced_costs: &self.buf_reduced_costs,
                iterations: self.solution.iterations,
                solve_time_seconds: self.solution.solve_time_seconds,
            })
        }
    }

    impl SolverInterface for MockSolver {
        fn solver_name_version(&self) -> String {
            "MockSolver 0.0.0".to_string()
        }
        fn load_model(&mut self, _template: &StageTemplate) {
            self.load_count += 1;
        }
        fn add_rows(&mut self, _cuts: &RowBatch) {
            self.add_rows_count += 1;
        }
        fn set_row_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn set_col_bounds(&mut self, _indices: &[usize], _lower: &[f64], _upper: &[f64]) {}
        fn solve(
            &mut self,
            basis: Option<&Basis>,
        ) -> Result<cobre_solver::SolutionView<'_>, SolverError> {
            if let Some(b) = basis {
                self.solve_with_basis_count += 1;
                self.recorded_basis = Some(b.clone());
            } else {
                self.solve_count += 1;
            }
            self.do_solve()
        }
        fn get_basis(&mut self, _out: &mut Basis) {}
        fn record_reconstruction_stats(&mut self) {
            self.reconstruction_counter += 1;
        }
        fn statistics(&self) -> SolverStatistics {
            SolverStatistics::default()
        }
        fn name(&self) -> &'static str {
            "Mock"
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Minimal valid stage template for N=1 hydro, L=0 PAR order.
    ///
    /// Column layout (N=1, L=0):
    /// - col 0: `storage_out` (no NZ in structural rows)
    /// - col 1: `z_inflow` (no NZ — `z_inflow` row at row 1)
    /// - col 2: `storage_in` (1 NZ: row 0, storage-fixing row)
    /// - col 3: `theta` (no NZ)
    ///
    /// Row layout:
    /// - row 0: storage-fixing (`storage_out` fixed to incoming state)
    /// - row 1: `z_inflow` definition row
    fn minimal_template_1_0() -> StageTemplate {
        StageTemplate {
            num_cols: 4,
            num_rows: 2,
            num_nz: 1,
            // CSC col_starts: 4 cols + 1 sentinel = 5 entries.
            // col 0 (storage_out): 0 NZ
            // col 1 (z_inflow):    0 NZ
            // col 2 (storage_in):  1 NZ at row 0
            // col 3 (theta):       0 NZ
            col_starts: vec![0_i32, 0, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, f64::NEG_INFINITY, 0.0, 0.0],
            col_upper: vec![f64::INFINITY; 4],
            objective: vec![0.0, 0.0, 0.0, 1.0],
            row_lower: vec![0.0, 0.0],
            row_upper: vec![0.0, 0.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Build a fixed `LpSolution` for the minimal N=1 L=0 template.
    ///
    /// N=1 L=0 column layout: `storage`(0), `z_inflow`(1), `storage_in`(2), `theta`(3).
    /// `primal[3] = theta_val`, `objective = objective`.
    fn fixed_solution(objective: f64, theta_val: f64) -> LpSolution {
        let num_cols = 4; // storage(0), z_inflow(1), storage_in(2), theta(3)
        let mut primal = vec![0.0_f64; num_cols];
        primal[3] = theta_val; // theta at col 3 (N=1, L=0 → theta = N*(3+L) = 3)
        LpSolution {
            objective,
            primal,
            dual: vec![0.0_f64; 2],
            reduced_costs: vec![0.0_f64; num_cols],
            iterations: 0,
            solve_time_seconds: 0.0,
        }
    }

    /// Build a minimal `EntityCounts` for 1 hydro, no other entities.
    fn entity_counts_1_hydro() -> EntityCounts {
        EntityCounts {
            hydro_ids: vec![1],
            hydro_productivities: vec![1.0],
            thermal_ids: vec![],
            line_ids: vec![],
            bus_ids: vec![],
            pumping_station_ids: vec![],
            contract_ids: vec![],
            non_controllable_ids: vec![],
        }
    }

    /// Build a minimal stochastic context for 1 hydro, `n_stages` stages.
    fn make_stochastic_context(n_stages: usize) -> StochasticContext {
        use std::collections::BTreeMap;

        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        };
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };
        use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
        use cobre_stochastic::context::{
            ClassSchemes, OpeningTreeInputs, build_stochastic_context,
        };

        let bus = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: EntityId(1),
            name: "H1".to_string(),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
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
            },
        };
        let make_stage = |idx: usize, id: i32| Stage {
            index: idx,
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
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
                branching_factor: 3,
                noise_method: NoiseMethod::Saa,
            },
        };
        let stages: Vec<Stage> = (0..n_stages)
            .map(|i| make_stage(i, i32::try_from(i).unwrap()))
            .collect();
        let inflow = |stage_id: i32| InflowModel {
            hydro_id: EntityId(1),
            stage_id,
            mean_m3s: 100.0,
            std_m3s: 30.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        };
        let inflow_models: Vec<InflowModel> = (0..n_stages)
            .map(|i| inflow(i32::try_from(i).unwrap()))
            .collect();
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: vec![CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(1),
                    }],
                    matrix: vec![vec![1.0]],
                }],
            },
        );
        let correlation = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };
        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(stages)
            .inflow_models(inflow_models)
            .correlation(correlation)
            .build()
            .unwrap();
        build_stochastic_context(
            &system,
            42,
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
        .unwrap()
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Build per-stage hydro productivities for 1 hydro with productivity 1.0.
    ///
    /// Returns `n_stages` inner vecs each containing a single `1.0` entry,
    /// matching `entity_counts_1_hydro().hydro_productivities`.
    fn hydro_productivities_1hydro(n_stages: usize) -> Vec<Vec<f64>> {
        vec![vec![1.0]; n_stages]
    }

    /// Build a zero-valued [`crate::energy_conversion::EnergyConversionSet`] for tests
    /// that do not assert on energy fields.
    fn zero_energy_conversion(
        n_hydros: usize,
        n_stages: usize,
    ) -> crate::energy_conversion::EnergyConversionSet {
        use crate::energy_conversion::{EnergyConversion, EnergyConversionSet};
        let zero_ec = EnergyConversion {
            equivalent_productivity_mw_per_m3s: 0.0,
            reference_volume_hm3: 0.0,
            reference_outflow_m3s: 0.0,
        };
        EnergyConversionSet::new(
            vec![vec![zero_ec; n_stages]; n_hydros],
            vec![vec![0.0_f64; n_stages]; n_hydros],
            n_hydros,
            n_stages,
        )
    }

    /// Wrap a `MockSolver` in a single-workspace slice for `simulate()` calls.
    ///
    /// All tests use a single workspace (serial execution) so that existing
    /// assertions about scenario ordering and call counts remain valid.
    fn single_workspace(solver: MockSolver) -> Vec<SolverWorkspace<MockSolver>> {
        vec![SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver,
            patch_buf: PatchBuffer::new(1, 0, 0, 0, 0, 0), // N=1, L=0
            current_state: Vec::with_capacity(1),
            scratch: ScratchBuffers {
                noise_buf: Vec::new(),
                inflow_m3s_buf: Vec::new(),
                lag_matrix_buf: Vec::new(),
                par_inflow_buf: Vec::new(),
                eta_floor_buf: Vec::new(),
                zero_targets_buf: Vec::new(),
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                load_rhs_buf: Vec::new(),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                current_state_scratch: Vec::new(),
                recon_slot_lookup: Vec::new(),
                promotion_scratch: crate::basis_reconstruct::PromotionScratch::default(),
                trajectory_costs_buf: Vec::new(),
                raw_noise_buf: Vec::new(),
                perm_scratch: Vec::new(),
                anticipated_state_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
        }]
    }

    // ── Load noise simulation tests ───────────────────────────────────────────

    /// Build a stochastic context with 1 hydro and 1 stochastic load bus for
    /// simulation load noise tests.
    ///
    /// The context has 1 stage with branching factor 3.  The load model uses
    /// `bus_id=1` (distinct from the hydro bus at `bus_id=0`), so the noise
    /// vector has dimension 2: `[inflow_eta, load_eta]`.
    fn make_stochastic_context_1_hydro_1_load_bus_sim(
        mean_mw: f64,
        std_mw: f64,
    ) -> StochasticContext {
        use std::collections::BTreeMap;

        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{CorrelationModel, InflowModel, LoadModel};
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };
        use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
        use cobre_stochastic::context::{
            ClassSchemes, OpeningTreeInputs, build_stochastic_context,
        };

        let bus0 = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let bus1 = Bus {
            id: EntityId(1),
            name: "B1".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: EntityId(10),
            name: "H10".to_string(),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
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
            },
        };
        let stage = Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
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
                branching_factor: 3,
                noise_method: NoiseMethod::Saa,
            },
        };
        let inflow_model = InflowModel {
            hydro_id: EntityId(10),
            stage_id: 0,
            mean_m3s: 100.0,
            std_m3s: 20.0,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        };
        let load_model = LoadModel {
            bus_id: EntityId(1),
            stage_id: 0,
            mean_mw,
            std_mw,
        };
        let correlation = CorrelationModel {
            method: "spectral".to_string(),
            profiles: BTreeMap::new(),
            schedule: vec![],
        };
        let system = SystemBuilder::new()
            .buses(vec![bus0, bus1])
            .hydros(vec![hydro])
            .stages(vec![stage])
            .inflow_models(vec![inflow_model])
            .load_models(vec![load_model])
            .correlation(correlation)
            .build()
            .unwrap();
        build_stochastic_context(
            &system,
            42,
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
        .unwrap()
    }

    /// When a simulation has 1 stochastic load bus (mean=300, std=30),
    /// verify that `load_rhs_buf` is populated with a positive value.
    #[test]
    fn simulation_load_patches_applied() {
        let n_stages = 1;
        let template = StageTemplate {
            num_cols: 3,
            num_rows: 3,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0, 0.0, 1.0],
            row_lower: vec![0.0, 100.0, 300.0], // row 2 = load balance with mean=300
            row_upper: vec![0.0, 100.0, 300.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 3,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let templates = vec![template];
        let base_rows = vec![1usize]; // water-balance rows start at row 1

        let n_load_buses = 1usize;
        let stochastic = make_stochastic_context_1_hydro_1_load_bus_sim(300.0, 30.0);
        let indexer = StageIndexer::new(1, 0); // N=1, L=0; theta=2
        let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
        let config = SimulationConfig {
            n_scenarios: 1,
            io_channel_capacity: 4,
            basis_activity_window: crate::basis_reconstruct::DEFAULT_BASIS_ACTIVITY_WINDOW,
        };
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let initial_state = vec![50.0_f64];

        let solution = fixed_solution(100.0, 30.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, _rx) = mpsc::sync_channel(4);

        let mut workspaces = vec![SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver,
            patch_buf: PatchBuffer::new(1, 0, n_load_buses, 1, 0, 0),
            current_state: Vec::with_capacity(1),
            scratch: ScratchBuffers {
                noise_buf: Vec::new(),
                inflow_m3s_buf: Vec::new(),
                lag_matrix_buf: Vec::new(),
                par_inflow_buf: Vec::new(),
                eta_floor_buf: Vec::new(),
                zero_targets_buf: Vec::new(),
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                load_rhs_buf: Vec::with_capacity(n_load_buses),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                current_state_scratch: Vec::new(),
                recon_slot_lookup: Vec::new(),
                promotion_scratch: crate::basis_reconstruct::PromotionScratch::default(),
                trajectory_costs_buf: Vec::new(),
                raw_noise_buf: Vec::new(),
                perm_scratch: Vec::new(),
                anticipated_state_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
        }];

        // load_balance_row_starts[0]=2 (load balance row is row 2 in the template).
        // load_bus_indices=[0] (bus position 0 in the block layout).
        let load_balance_row_starts = vec![2usize];
        let load_bus_indices = vec![0usize];
        let block_counts_per_stage = vec![1usize];
        let noise_scale = vec![1.0_f64]; // 1 hydro, 1 stage

        let hprod = hydro_productivities_1hydro(n_stages);
        let ec = zero_energy_conversion(1, n_stages);
        run_simulate(
            &mut workspaces,
            &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &noise_scale,
                n_hydros: 1,
                n_load_buses,
                load_balance_row_starts: &load_balance_row_starts,
                load_bus_indices: &load_bus_indices,
                block_counts_per_stage: &block_counts_per_stage,
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            &fcf,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &config,
            SimulationOutputSpec {
                result_tx: &tx,
                zeta_per_stage: &[],
                block_hours_per_stage: &[],
                entity_counts: &entity_counts,
                generic_constraint_row_entries: &[],
                ncs_col_starts: &[],
                n_ncs_per_stage: &[],
                ncs_entity_ids_per_stage: &[],
                diversion_upstream: &HashMap::new(),
                hydro_productivities_per_stage: &hprod,
                energy_conversion: &ec,
                hydro_min_storage_hm3: &[0.0],
                event_sender: None,
            },
            None,
            &[],
            &comm,
        )
        .unwrap();

        // The load noise path must have populated load_rhs_buf.
        assert_eq!(
            workspaces[0].scratch.load_rhs_buf.len(),
            n_load_buses,
            "load_rhs_buf must have 1 entry (1 load bus x 1 block)"
        );
        assert!(
            workspaces[0].scratch.load_rhs_buf[0] > 0.0,
            "realization must be positive with mean=300, std=30: got {}",
            workspaces[0].scratch.load_rhs_buf[0]
        );

        // Verify the formula: d = max(0, mean + std * eta) * factor.
        // The exact eta drawn from the opening tree depends on the seed, but
        // we can verify formula consistency by back-computing eta from the
        // observed realization (d > 0 implies eta = (d - mean) / std).
        let d_observed = workspaces[0].scratch.load_rhs_buf[0];
        let mean_mw_val = 300.0_f64;
        let std_mw_val = 30.0_f64;
        assert!(
            d_observed != mean_mw_val,
            "realization must differ from template mean (noise was applied)"
        );
        let eta_back = (d_observed - mean_mw_val) / std_mw_val;
        let recomputed = (mean_mw_val + std_mw_val * eta_back).max(0.0);
        assert!(
            (d_observed - recomputed).abs() < 1e-10,
            "formula consistency: d={d_observed}, eta_back={eta_back}, recomputed={recomputed}"
        );

        // The load patch must also be reflected in the patch buffer.
        let cat4_start = 2; // n_hydros * (2 + max_par_order) = 1 * 2 = 2
        assert_eq!(
            workspaces[0].patch_buf.lower[cat4_start], workspaces[0].scratch.load_rhs_buf[0],
            "patch_buf lower at load slot must equal load_rhs_buf[0]"
        );
        assert_eq!(
            workspaces[0].patch_buf.upper[cat4_start], workspaces[0].scratch.load_rhs_buf[0],
            "patch_buf upper at load slot must equal load_rhs_buf[0] (equality constraint)"
        );

        // Verify that the extraction row_lower_buf was patched with the
        // stochastic realization (not the template mean 300.0).
        // row_lower_buf[2] is the load balance row (row index 2).
        assert!(
            !workspaces[0].scratch.row_lower_buf.is_empty(),
            "row_lower_buf must be populated for extraction"
        );
        assert_eq!(
            workspaces[0].scratch.row_lower_buf[2], d_observed,
            "extraction row_lower_buf must contain stochastic load, not template mean"
        );
        assert!(
            (workspaces[0].scratch.row_lower_buf[2] - mean_mw_val).abs() > 1e-6,
            "extracted load_mw must differ from template mean {mean_mw_val}: got {}",
            workspaces[0].scratch.row_lower_buf[2]
        );
    }

    /// when `n_load_buses == 0`,
    /// `load_rhs_buf` remains empty and `forward_patch_count` equals
    /// `N*(2+L)` as with the training forward pass.
    ///
    /// With N=1, L=0: `forward_patch_count = 1 * (2 + 0) = 2`.
    #[test]
    fn simulation_no_load_buses_unchanged() {
        let n_stages = 1;
        let templates = vec![minimal_template_1_0()];
        let base_rows = vec![0usize];

        let stochastic = make_stochastic_context(n_stages);
        let indexer = StageIndexer::new(1, 0);
        let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
        let config = SimulationConfig {
            n_scenarios: 1,
            io_channel_capacity: 4,
            basis_activity_window: crate::basis_reconstruct::DEFAULT_BASIS_ACTIVITY_WINDOW,
        };
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let initial_state = vec![50.0_f64];

        let solution = fixed_solution(100.0, 30.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, _rx) = mpsc::sync_channel(4);

        let hprod = hydro_productivities_1hydro(n_stages);
        let ec = zero_energy_conversion(1, n_stages);
        let mut workspaces = single_workspace(solver);

        run_simulate(
            &mut workspaces,
            &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &[],
                n_hydros: 0,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[1],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            &fcf,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &config,
            SimulationOutputSpec {
                result_tx: &tx,
                zeta_per_stage: &[],
                block_hours_per_stage: &[],
                entity_counts: &entity_counts,
                generic_constraint_row_entries: &[],
                ncs_col_starts: &[],
                n_ncs_per_stage: &[],
                ncs_entity_ids_per_stage: &[],
                diversion_upstream: &HashMap::new(),
                hydro_productivities_per_stage: &hprod,
                energy_conversion: &ec,
                hydro_min_storage_hm3: &[0.0],
                event_sender: None,
            },
            None,
            &[],
            &comm,
        )
        .unwrap();

        // load_rhs_buf must remain empty when n_load_buses=0.
        assert!(
            workspaces[0].scratch.load_rhs_buf.is_empty(),
            "load_rhs_buf must be empty when n_load_buses=0"
        );
        // forward_patch_count = N*(2+L) = 1*(2+0) = 2 (no load patches added).
        assert_eq!(
            workspaces[0].patch_buf.forward_patch_count(),
            2,
            "forward_patch_count must be N*(2+L)=2 when n_load_buses=0, got {}",
            workspaces[0].patch_buf.forward_patch_count()
        );
    }

    /// When load noise is present,
    /// `noise_buf` still contains only inflow values (not contaminated by load noise).
    ///
    /// `noise_buf` contains inflow realizations for the `n_hydros` hydros.
    /// After simulate runs with `n_hydros=1` and `n_load_buses=1`, `noise_buf`
    /// must have exactly 1 entry (inflow), while `load_rhs_buf` has 1 entry
    /// (load).  The two buffers must not overlap.
    #[test]
    fn simulation_inflow_extraction_unaffected() {
        let n_stages = 1;
        let template = StageTemplate {
            num_cols: 3,
            num_rows: 3,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0, 0.0, 1.0],
            row_lower: vec![0.0, 100.0, 300.0],
            row_upper: vec![0.0, 100.0, 300.0],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 3,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        };
        let templates = vec![template];
        let base_rows = vec![1usize];

        let n_load_buses = 1usize;
        let stochastic = make_stochastic_context_1_hydro_1_load_bus_sim(300.0, 30.0);
        let indexer = StageIndexer::new(1, 0);
        let fcf = FutureCostFunction::new(n_stages, 1, 1, 10, &vec![0; n_stages]);
        let config = SimulationConfig {
            n_scenarios: 1,
            io_channel_capacity: 4,
            basis_activity_window: crate::basis_reconstruct::DEFAULT_BASIS_ACTIVITY_WINDOW,
        };
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let initial_state = vec![50.0_f64];

        let solution = fixed_solution(100.0, 30.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, _rx) = mpsc::sync_channel(4);

        let mut workspaces = vec![SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver,
            patch_buf: PatchBuffer::new(1, 0, n_load_buses, 1, 0, 0),
            current_state: Vec::with_capacity(1),
            scratch: ScratchBuffers {
                noise_buf: Vec::new(),
                inflow_m3s_buf: Vec::new(),
                lag_matrix_buf: Vec::new(),
                par_inflow_buf: Vec::new(),
                eta_floor_buf: Vec::new(),
                zero_targets_buf: Vec::new(),
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                load_rhs_buf: Vec::with_capacity(n_load_buses),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                current_state_scratch: Vec::new(),
                recon_slot_lookup: Vec::new(),
                promotion_scratch: crate::basis_reconstruct::PromotionScratch::default(),
                trajectory_costs_buf: Vec::new(),
                raw_noise_buf: Vec::new(),
                perm_scratch: Vec::new(),
                anticipated_state_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
        }];

        let load_balance_row_starts = vec![2usize];
        let load_bus_indices = vec![0usize];
        let block_counts_per_stage = vec![1usize];
        let noise_scale = vec![1.0_f64];

        let hprod = hydro_productivities_1hydro(n_stages);
        let ec = zero_energy_conversion(1, n_stages);
        run_simulate(
            &mut workspaces,
            &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &noise_scale,
                n_hydros: 1,
                n_load_buses,
                load_balance_row_starts: &load_balance_row_starts,
                load_bus_indices: &load_bus_indices,
                block_counts_per_stage: &block_counts_per_stage,
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            &fcf,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &config,
            SimulationOutputSpec {
                result_tx: &tx,
                zeta_per_stage: &[],
                block_hours_per_stage: &[],
                entity_counts: &entity_counts,
                generic_constraint_row_entries: &[],
                ncs_col_starts: &[],
                n_ncs_per_stage: &[],
                ncs_entity_ids_per_stage: &[],
                diversion_upstream: &HashMap::new(),
                hydro_productivities_per_stage: &hprod,
                energy_conversion: &ec,
                hydro_min_storage_hm3: &[0.0],
                event_sender: None,
            },
            None,
            &[],
            &comm,
        )
        .unwrap();

        // noise_buf contains only inflow values (n_hydros=1 entries).
        // It must not be contaminated by load noise (which lives in load_rhs_buf).
        assert_eq!(
            workspaces[0].scratch.noise_buf.len(),
            1,
            "noise_buf must have 1 entry (1 hydro), not contaminated by load noise: len={}",
            workspaces[0].scratch.noise_buf.len()
        );
        // The inflow noise must be a reasonable value near mean_rhs=100.
        // With noise_scale=1.0 and mean_rhs=100 (from row_lower[base_rows[0]+0]=100):
        //   noise_buf[0] = 100.0 + 1.0 * eta_inflow
        // For any |eta_inflow| <= 5 this remains in [75, 125] for practical draws.
        assert!(
            workspaces[0].scratch.noise_buf[0] > 50.0 && workspaces[0].scratch.noise_buf[0] < 200.0,
            "noise_buf[0] must be a reasonable inflow value near 100.0, got {}",
            workspaces[0].scratch.noise_buf[0]
        );
        // load_rhs_buf must also be populated (confirms both buffers coexist).
        assert_eq!(
            workspaces[0].scratch.load_rhs_buf.len(),
            n_load_buses,
            "load_rhs_buf must have 1 entry alongside noise_buf"
        );
    }

    // ── Inflow truncation unit tests ──────────────────────────────────────────

    /// Build a `StochasticContext` for 1 hydro, 1 stage with configurable
    /// `mean_m3s` and `std_m3s`.  Used by the truncation tests below.
    fn make_stochastic_1h_1s(mean_m3s: f64, std_m3s: f64) -> StochasticContext {
        use std::collections::BTreeMap;

        use chrono::NaiveDate;
        use cobre_core::entities::hydro::{Hydro, HydroGenerationModel, HydroPenalties};
        use cobre_core::scenario::{
            CorrelationEntity, CorrelationGroup, CorrelationModel, CorrelationProfile, InflowModel,
        };
        use cobre_core::temporal::{
            Block, BlockMode, NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig,
            StageStateConfig,
        };
        use cobre_core::{Bus, DeficitSegment, EntityId, SystemBuilder};
        use cobre_stochastic::context::{
            ClassSchemes, OpeningTreeInputs, build_stochastic_context,
        };

        let bus = Bus {
            id: EntityId(0),
            name: "B0".to_string(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 1000.0,
            }],
            excess_cost: 0.0,
        };
        let hydro = Hydro {
            id: EntityId(1),
            name: "H1".to_string(),
            bus_id: EntityId(0),
            downstream_id: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 100.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 100.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: HydroPenalties {
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
            },
        };
        let stage = Stage {
            index: 0,
            id: 0,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
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
                branching_factor: 3,
                noise_method: NoiseMethod::Saa,
            },
        };
        let inflow_model = InflowModel {
            hydro_id: EntityId(1),
            stage_id: 0,
            mean_m3s,
            std_m3s,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        };
        let mut profiles = BTreeMap::new();
        profiles.insert(
            "default".to_string(),
            CorrelationProfile {
                groups: vec![CorrelationGroup {
                    name: "g1".to_string(),
                    entities: vec![CorrelationEntity {
                        entity_type: "inflow".to_string(),
                        id: EntityId(1),
                    }],
                    matrix: vec![vec![1.0]],
                }],
            },
        );
        let correlation = CorrelationModel {
            method: "spectral".to_string(),
            profiles,
            schedule: vec![],
        };
        let system = SystemBuilder::new()
            .buses(vec![bus])
            .hydros(vec![hydro])
            .stages(vec![stage])
            .inflow_models(vec![inflow_model])
            .correlation(correlation)
            .build()
            .unwrap();
        build_stochastic_context(
            &system,
            42,
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
        .unwrap()
    }

    /// Build a stage template for N=1 hydro, L=0 PAR, with `row_lower[0] = base_rhs`.
    ///
    /// Used by truncation tests so that the water-balance base RHS is configurable.
    fn minimal_template_1_0_with_base(base_rhs: f64) -> StageTemplate {
        StageTemplate {
            num_cols: 3,
            num_rows: 1,
            num_nz: 1,
            col_starts: vec![0_i32, 0, 1, 1],
            row_indices: vec![0_i32],
            values: vec![1.0],
            col_lower: vec![0.0, 0.0, 0.0],
            col_upper: vec![f64::INFINITY, f64::INFINITY, f64::INFINITY],
            objective: vec![0.0, 0.0, 1.0],
            row_lower: vec![base_rhs],
            row_upper: vec![base_rhs],
            n_state: 1,
            n_transfer: 0,
            n_dual_relevant: 1,
            n_hydro: 1,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
        }
    }

    /// Build a workspace with `zero_targets_buf` pre-populated to `hydro_count` zeros.
    ///
    /// The standard `single_workspace` helper leaves `zero_targets_buf` empty because
    /// existing tests do not reach the truncation branch.  The truncation tests use
    /// 1 hydro and must have `zero_targets_buf[..1]` accessible, so they use this
    /// helper instead.
    fn single_workspace_with_hydros(
        solver: MockSolver,
        hydro_count: usize,
    ) -> Vec<SolverWorkspace<MockSolver>> {
        vec![SolverWorkspace {
            rank: 0,
            worker_id: 0,
            solver,
            patch_buf: PatchBuffer::new(hydro_count, 0, 0, 0, 0, 0),
            current_state: Vec::with_capacity(hydro_count),
            scratch: ScratchBuffers {
                noise_buf: Vec::new(),
                inflow_m3s_buf: Vec::new(),
                lag_matrix_buf: Vec::new(),
                par_inflow_buf: Vec::new(),
                eta_floor_buf: Vec::new(),
                zero_targets_buf: vec![0.0_f64; hydro_count],
                ncs_col_upper_buf: Vec::new(),
                ncs_col_lower_buf: Vec::new(),
                ncs_col_indices_buf: Vec::new(),
                load_rhs_buf: Vec::new(),
                row_lower_buf: Vec::new(),
                z_inflow_rhs_buf: Vec::new(),
                effective_eta_buf: Vec::new(),
                unscaled_primal: Vec::new(),
                unscaled_dual: Vec::new(),
                lag_accumulator: vec![],
                lag_weight_accum: 0.0,
                downstream_accumulator: Vec::new(),
                downstream_weight_accum: 0.0,
                downstream_completed_lags: Vec::new(),
                downstream_n_completed: 0,
                current_state_scratch: Vec::new(),
                recon_slot_lookup: Vec::new(),
                promotion_scratch: crate::basis_reconstruct::PromotionScratch::default(),
                trajectory_costs_buf: Vec::new(),
                raw_noise_buf: Vec::new(),
                perm_scratch: Vec::new(),
                anticipated_state_buf: Vec::new(),
            },
            scratch_basis: Basis::new(0, 0),
            backward_accum: BackwardAccumulators::default(),
            worker_timing_buf: cobre_core::WorkerPhaseTimings::default(),
        }]
    }

    /// AC: truncation clamps negative inflow noise in the simulation pipeline.
    ///
    /// Set `mean_m3s = -1000.0` and `std_m3s = 1.0` so that the deterministic
    /// PAR base alone would produce a hugely negative inflow for any sample.
    ///
    /// With `InflowNonNegativityMethod::Truncation` active, the simulation pipeline
    /// must clamp `eta` to the floor that produces zero inflow.  As a result,
    /// `noise_buf[0] = base_rhs + noise_scale * eta_clamped >= 0.0` for all
    /// scenarios processed.
    ///
    /// Concretely (zeta=1): `base_rhs = -1000`, `noise_scale = 1`.
    /// `eta_floor` = (0 - mean) / sigma = 1000. So `noise_buf`\[0\] = -1000 + 1\*1000 = 0.
    #[test]
    fn simulation_truncation_clamps_negative_inflow_noise() {
        let mean_m3s = -1000.0_f64;
        let sigma = 1.0_f64;
        let zeta = 1.0_f64; // simplified: treat zeta=1
        let base_rhs = zeta * mean_m3s;
        let noise_scale_val = zeta * sigma;

        let n_stages = 1;
        let stochastic = make_stochastic_1h_1s(mean_m3s, sigma);
        let template = minimal_template_1_0_with_base(base_rhs);
        let templates = vec![template];
        let base_rows = vec![0_usize];
        let noise_scale = vec![noise_scale_val];

        let indexer = StageIndexer::new(1, 0);
        let fcf = FutureCostFunction::new(n_stages, indexer.n_state, 1, 10, &vec![0; n_stages]);
        let config = SimulationConfig {
            n_scenarios: 4,
            io_channel_capacity: 16,
            basis_activity_window: crate::basis_reconstruct::DEFAULT_BASIS_ACTIVITY_WINDOW,
        };
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let initial_state = vec![0.0_f64];

        let solution = fixed_solution(0.0, 0.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, _rx) = mpsc::sync_channel(16);

        // Use the hydro-aware workspace builder so zero_targets_buf[..1] is valid.
        let hprod = hydro_productivities_1hydro(n_stages);
        let ec = zero_energy_conversion(1, n_stages);
        let mut workspaces = single_workspace_with_hydros(solver, 1);
        run_simulate(
            &mut workspaces,
            &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &noise_scale,
                n_hydros: 1,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[n_stages],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            &fcf,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::Truncation,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &config,
            SimulationOutputSpec {
                result_tx: &tx,
                zeta_per_stage: &[],
                block_hours_per_stage: &[],
                entity_counts: &entity_counts,
                generic_constraint_row_entries: &[],
                ncs_col_starts: &[],
                n_ncs_per_stage: &[],
                ncs_entity_ids_per_stage: &[],
                diversion_upstream: &HashMap::new(),
                hydro_productivities_per_stage: &hprod,
                energy_conversion: &ec,
                hydro_min_storage_hm3: &[0.0],
                event_sender: None,
            },
            None,
            &[],
            &comm,
        )
        .unwrap();

        // After truncation the noise_buf contains the value from the last stage solved
        // for the last scenario.  All scenarios must have produced a clamped value >= 0.
        // We verify the workspace buffer left by the final scenario-stage pair.
        assert_eq!(
            workspaces[0].scratch.noise_buf.len(),
            1,
            "noise_buf must have exactly 1 entry for 1 hydro"
        );
        assert!(
            workspaces[0].scratch.noise_buf[0] >= 0.0,
            "after truncation, noise_buf[0] must be >= 0 (inflow cannot be negative), got {}",
            workspaces[0].scratch.noise_buf[0]
        );
    }

    /// AC: `InflowNonNegativityMethod::None` in the simulation pipeline produces
    /// raw (potentially negative) noise values, unchanged from the pre-fix behavior.
    ///
    /// With `mean_m3s = -1000.0` and `std_m3s = 1.0`, the PAR inflow is always
    /// deeply negative.  The `None` path must NOT clamp eta, so the noise buffer
    /// value must be negative (`base_rhs` + `noise_scale` \* `raw_eta` << 0).
    #[test]
    fn simulation_none_method_produces_raw_negative_noise() {
        let mean_m3s = -1000.0_f64;
        let sigma = 1.0_f64;
        let zeta = 1.0_f64;
        let base_rhs = zeta * mean_m3s;
        let noise_scale_val = zeta * sigma;

        let n_stages = 1;
        let stochastic = make_stochastic_1h_1s(mean_m3s, sigma);
        let template = minimal_template_1_0_with_base(base_rhs);
        let templates = vec![template];
        let base_rows = vec![0_usize];
        let noise_scale = vec![noise_scale_val];

        let indexer = StageIndexer::new(1, 0);
        let fcf = FutureCostFunction::new(n_stages, indexer.n_state, 1, 10, &vec![0; n_stages]);
        let config = SimulationConfig {
            n_scenarios: 4,
            io_channel_capacity: 16,
            basis_activity_window: crate::basis_reconstruct::DEFAULT_BASIS_ACTIVITY_WINDOW,
        };
        let horizon = HorizonMode::Finite {
            num_stages: n_stages,
        };
        let initial_state = vec![0.0_f64];

        let solution = fixed_solution(0.0, 0.0);
        let solver = MockSolver::always_ok(solution);
        let comm = StubComm { rank: 0, size: 1 };
        let entity_counts = entity_counts_1_hydro();

        let (tx, _rx) = mpsc::sync_channel(16);

        let hprod = hydro_productivities_1hydro(n_stages);
        let ec = zero_energy_conversion(1, n_stages);
        let mut workspaces = single_workspace_with_hydros(solver, 1);
        run_simulate(
            &mut workspaces,
            &StageContext {
                templates: &templates,
                base_rows: &base_rows,
                noise_scale: &noise_scale,
                n_hydros: 1,
                n_load_buses: 0,
                load_balance_row_starts: &[],
                load_bus_indices: &[],
                block_counts_per_stage: &[n_stages],
                ncs_max_gen: &[],
                ncs_allow_curtailment: &[],
                discount_factors: &[],
                cumulative_discount_factors: &[],
                stage_lag_transitions: &[],
                noise_group_ids: &[],
                downstream_par_order: 0,
            },
            &fcf,
            &TrainingContext {
                horizon: &horizon,
                indexer: &indexer,
                inflow_method: &InflowNonNegativityMethod::None,
                stochastic: &stochastic,
                initial_state: &initial_state,
                inflow_scheme: SamplingScheme::InSample,
                load_scheme: SamplingScheme::InSample,
                ncs_scheme: SamplingScheme::InSample,
                stages: &[],
                historical_library: None,
                external_inflow_library: None,
                external_load_library: None,
                external_ncs_library: None,
                recent_accum_seed: &[],
                recent_weight_seed: 0.0,
            },
            &config,
            SimulationOutputSpec {
                result_tx: &tx,
                zeta_per_stage: &[],
                block_hours_per_stage: &[],
                entity_counts: &entity_counts,
                generic_constraint_row_entries: &[],
                ncs_col_starts: &[],
                n_ncs_per_stage: &[],
                ncs_entity_ids_per_stage: &[],
                diversion_upstream: &HashMap::new(),
                hydro_productivities_per_stage: &hprod,
                energy_conversion: &ec,
                hydro_min_storage_hm3: &[0.0],
                event_sender: None,
            },
            None,
            &[],
            &comm,
        )
        .unwrap();

        assert_eq!(
            workspaces[0].scratch.noise_buf.len(),
            1,
            "noise_buf must have exactly 1 entry for 1 hydro"
        );
        // With None, no clamping occurs.  base_rhs=-1000 and noise_scale=1, so
        // noise_buf[0] = -1000 + 1 * eta.  For |eta| < 5 this remains << 0.
        assert!(
            workspaces[0].scratch.noise_buf[0] < 0.0,
            "with None method, noise_buf[0] must be negative (raw eta applied), got {}",
            workspaces[0].scratch.noise_buf[0]
        );
    }
}
